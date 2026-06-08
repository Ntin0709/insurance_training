"""Build a local RAG index manifest over scraped insurance PDF chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insurance_agent.rag_index import build_index_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/insurance/pdf_source_chunks.jsonl")
    parser.add_argument("--manifest", default="data/insurance/rag_index_manifest.json")
    args = parser.parse_args()

    manifest = build_index_manifest(args.chunks, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
