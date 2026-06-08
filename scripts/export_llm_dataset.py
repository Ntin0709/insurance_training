"""Export banking tool-use SFT examples as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking_rl_env.llm import all_tool_schemas, export_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/banking_tool_sft.jsonl")
    parser.add_argument("--schemas-output", default="data/tool_schemas.json")
    args = parser.parse_args()

    examples = export_dataset(args.output)
    schema_path = Path(args.schemas_output)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(all_tool_schemas(), indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_examples={len(examples)} output={args.output}")
    print(f"wrote_tool_schemas={len(all_tool_schemas())} output={args.schemas_output}")


if __name__ == "__main__":
    main()
