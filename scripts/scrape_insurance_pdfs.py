"""Download/extract discovered official insurance PDFs into source docs and chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.models import SourceDocument
from insurance_agent.scraper import chunk_documents, fetch_url, pdf_to_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="data/insurance/pdf_inventory.json")
    parser.add_argument("--output-dir", default="data/insurance/pdf_sources")
    parser.add_argument("--chunks-output", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing; 0 means all PDFs.")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    urls = inventory["pdf_urls"]
    if args.limit:
        urls = urls[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    failures = []

    for index, url in enumerate(urls, start=1):
        source_id = _source_id(url, index)
        existing_path = output_dir / f"{source_id}.json"
        if args.resume and existing_path.exists():
            document = SourceDocument(**json.loads(existing_path.read_text(encoding="utf-8")))
            documents.append(document)
            print(f"skip {index}/{len(urls)} {source_id} chars={len(document.text)}")
            continue
        try:
            raw = fetch_url(url, timeout=args.timeout)
            text = pdf_to_text(raw)
            document = SourceDocument(
                source_id=source_id,
                url=url,
                title=_title_from_url(url),
                document_type="pdf",
                text=text,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )
            documents.append(document)
            Path(output_dir, f"{source_id}.json").write_text(
                json.dumps(document.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"ok {index}/{len(urls)} {source_id} chars={len(text)}")
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"fail {index}/{len(urls)} {url} error={exc}")

    chunks = chunk_documents(documents)
    chunks_output = Path(args.chunks_output)
    chunks_output.parent.mkdir(parents=True, exist_ok=True)
    with chunks_output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, sort_keys=True) + "\n")

    failure_path = output_dir / "_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")

    print(f"downloaded_pdfs={len(documents)}")
    print(f"failed_pdfs={len(failures)}")
    print(f"wrote_chunks={len(chunks)} output={args.chunks_output}")


def _source_id(url: str, index: int) -> str:
    stem = _title_from_url(url).lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"pdf_{index:04d}_{stem[:80]}"


def _title_from_url(url: str) -> str:
    tail = url.split("?")[0].rstrip("/").split("/")[-1]
    if tail.lower().endswith(".pdf"):
        tail = tail[:-4]
    return tail or "insurance_pdf"


if __name__ == "__main__":
    main()
