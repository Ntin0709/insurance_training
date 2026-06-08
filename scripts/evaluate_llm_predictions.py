"""Evaluate predicted tool-call sequences against oracle sequences.

Input JSONL format:
{"scenario_id": "book_fd", "predicted_actions": [{"tool": "AUTHENTICATE", "arguments": {}}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking_rl_env.llm import evaluate_action_sequence
from banking_rl_env.scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    scenarios = {scenario.scenario_id: scenario for scenario in SCENARIOS}
    total = 0
    exact = 0
    tool_seq = 0
    failures = []

    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result = evaluate_action_sequence(scenarios[row["scenario_id"]], row["predicted_actions"])
            total += 1
            exact += int(result["exact_sequence"])
            tool_seq += int(result["tool_sequence_accuracy"])
            if not result["exact_sequence"]:
                failures.append(result)

    print(
        json.dumps(
            {
                "total": total,
                "exact_sequence_accuracy": exact / total if total else 0,
                "tool_sequence_accuracy": tool_seq / total if total else 0,
                "failures": failures[:10],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
