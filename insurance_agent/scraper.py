"""Official-source scraper for insurance product knowledge."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from insurance_agent.models import SourceDocument


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return clean_text(" ".join(self.parts))


class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in {"a", "link"}:
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.links.add(normalize_url(urljoin(self.base_url, href)))


def load_source_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scrape_sources(config_path: str | Path, output_dir: str | Path) -> list[SourceDocument]:
    config = load_source_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for source in config["sources"]:
        raw = fetch_url(source["url"])
        if source["document_type"] == "html":
            text = html_to_text(raw.decode("utf-8", errors="ignore"))
        elif source["document_type"] == "pdf":
            text = pdf_to_text(raw)
        else:
            text = clean_text(raw.decode("utf-8", errors="ignore"))

        document = SourceDocument(
            source_id=source["source_id"],
            url=source["url"],
            title=source["title"],
            document_type=source["document_type"],
            text=text,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        documents.append(document)
        Path(output, f"{document.source_id}.json").write_text(
            json.dumps(document.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return documents


def fetch_url(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": "insurance-advisor-research/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_url(url: str) -> str:
    return urldefrag(url)[0].strip()


def same_registered_domain(url: str, allowed_domains: set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def extract_links(html: str, base_url: str) -> set[str]:
    parser = LinkExtractor(base_url)
    parser.feed(html)
    return parser.links


def discover_pdf_urls(
    start_urls: Iterable[str],
    allowed_domains: set[str],
    max_pages: int = 500,
    timeout: int = 8,
    verbose: bool = False,
) -> tuple[set[str], set[str]]:
    visited: set[str] = set()
    queued: list[str] = [normalize_url(url) for url in start_urls]
    pdf_urls: set[str] = set()

    while queued and len(visited) < max_pages:
        url = queued.pop(0)
        if url in visited or not same_registered_domain(url, allowed_domains):
            continue
        visited.add(url)
        if verbose and (len(visited) == 1 or len(visited) % 10 == 0):
            print(f"visited={len(visited)} queued={len(queued)} pdfs={len(pdf_urls)} url={url}", flush=True)
        try:
            raw = fetch_url(url, timeout=timeout)
        except Exception:
            continue

        lower_url = url.lower()
        if lower_url.endswith(".pdf"):
            pdf_urls.add(url)
            continue

        html = raw.decode("utf-8", errors="ignore")
        for link in extract_links(html, url):
            if not same_registered_domain(link, allowed_domains):
                continue
            if link.lower().endswith(".pdf") or ".pdf?" in link.lower():
                pdf_urls.add(link)
            elif _looks_like_crawlable_html(link):
                queued.append(link)

    return pdf_urls, visited


def sitemap_urls(root_url: str) -> set[str]:
    candidates = {
        urljoin(root_url, "/sitemap.xml"),
        urljoin(root_url, "/sitemap_index.xml"),
    }
    found: set[str] = set()
    for sitemap in candidates:
        try:
            text = fetch_url(sitemap, timeout=8).decode("utf-8", errors="ignore")
        except Exception:
            continue
        found.update(re.findall(r"<loc>\s*([^<]+)\s*</loc>", text, flags=re.IGNORECASE))
    return {normalize_url(url) for url in found}


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def pdf_to_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
        from io import BytesIO
    except ImportError as exc:
        raise RuntimeError("PDF scraping requires pypdf. Install with: python3 -m pip install pypdf") from exc

    reader = PdfReader(BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_crawlable_html(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    blocked_ext = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".svg",
        ".css",
        ".js",
        ".zip",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
    )
    return not path.endswith(blocked_ext)


def chunk_documents(documents: Iterable[SourceDocument], max_chars: int = 1800) -> list[dict]:
    chunks = []
    for document in documents:
        words = document.text.split()
        current: list[str] = []
        for word in words:
            if sum(len(item) + 1 for item in current) + len(word) > max_chars and current:
                chunks.append(_chunk(document, " ".join(current), len(chunks)))
                current = []
            current.append(word)
        if current:
            chunks.append(_chunk(document, " ".join(current), len(chunks)))
    return chunks


def _chunk(document: SourceDocument, text: str, index: int) -> dict:
    return {
        "chunk_id": f"{document.source_id}:{index}",
        "source_id": document.source_id,
        "title": document.title,
        "url": document.url,
        "text": text,
    }
