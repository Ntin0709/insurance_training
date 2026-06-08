"""Small command line utilities for evaluating the environment."""

from __future__ import annotations

import argparse
import json

from banking_rl_env.env import BankingToolEnv
from banking_rl_env.policies import oracle_action, run_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate banking RL environment policies.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dump-trace", action="store_true")
    args = parser.parse_args()

    env = BankingToolEnv(seed=args.seed)
    metrics = run_policy(env, oracle_action, episodes=args.episodes)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.dump_trace:
        env.reset(seed=args.seed)
        while True:
            _, _, terminated, truncated, _ = env.step(oracle_action(env))
            if terminated or truncated:
                break
        print(json.dumps(env.trace, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
