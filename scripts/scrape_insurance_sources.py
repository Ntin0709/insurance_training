"""Scrape official insurance source pages/PDFs into local JSON documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.scraper import chunk_documents, scrape_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="insurance_agent/config/sbi_general_sources.json")
    parser.add_argument("--output-dir", default="data/insurance/sources")
    parser.add_argument("--chunks-output", default="data/insurance/source_chunks.jsonl")
    args = parser.parse_args()

    documents = scrape_sources(args.config, args.output_dir)
    chunks = chunk_documents(documents)
    output = Path(args.chunks_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, sort_keys=True) + "\n")

    print(f"scraped_documents={len(documents)} output_dir={args.output_dir}")
    print(f"wrote_chunks={len(chunks)} output={args.chunks_output}")


if __name__ == "__main__":
    main()
