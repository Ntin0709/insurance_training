"""Discover official SBI General PDF URLs from pages and sitemaps."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.scraper import discover_pdf_urls, sitemap_urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="https://www.sbigeneral.in/")
    parser.add_argument("--allowed-domain", action="append", default=["sbigeneral.in"])
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="data/insurance/pdf_inventory.json")
    args = parser.parse_args()

    start_urls = {args.root}
    start_urls.update(sitemap_urls(args.root))
    pdf_urls, visited = discover_pdf_urls(
        start_urls,
        set(args.allowed_domain),
        max_pages=args.max_pages,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    inventory = {
        "root": args.root,
        "allowed_domains": args.allowed_domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "visited_pages": sorted(visited),
        "pdf_urls": sorted(pdf_urls),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")

    print(f"visited_pages={len(visited)}")
    print(f"pdf_urls={len(pdf_urls)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
