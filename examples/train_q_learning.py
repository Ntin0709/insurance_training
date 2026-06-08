"""Train a small tabular Q-learning baseline.

This is intentionally dependency-free. It is a sanity baseline for the
environment dynamics, not a final production agent.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking_rl_env import ACTIONS, BankingToolEnv, oracle_action


def obs_key(obs: dict) -> tuple:
    return (
        obs["scenario"],
        obs["target_tool"],
        obs["authenticated"],
        obs["confirmed"],
        tuple(obs["called_tools"]),
        tuple(obs["satisfied_prerequisites"]),
        obs["last_error_code"],
        obs["steps_remaining"],
    )


def greedy_action(q_table: dict[tuple, list[float]], state: tuple) -> int:
    values = q_table[state]
    return max(range(len(values)), key=values.__getitem__)


def evaluate(q_table: dict[tuple, list[float]], episodes: int, seed: int) -> dict[str, float]:
    env = BankingToolEnv(seed=seed, max_steps=10)
    total_reward = 0.0
    successes = 0
    total_steps = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            action = greedy_action(q_table, obs_key(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                successes += int(info["last_event"] == "success")
                total_steps += env.step_count
                break
        total_reward += episode_reward
    return {
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_length": total_steps / episodes,
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    rng = random.Random(args.seed)
    env = BankingToolEnv(seed=args.seed, max_steps=10)
    q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))

    for scenario in env.scenarios:
        for _ in range(args.oracle_epochs):
            obs, _ = env.reset(options={"scenario_id": scenario.scenario_id})
            while True:
                state = obs_key(obs)
                action = oracle_action(env)
                obs, reward, terminated, truncated, _ = env.step(action)
                q_table[state][action] += args.oracle_bonus + max(reward, 0.0)
                if terminated or truncated:
                    break

    epsilon = args.epsilon
    for episode in range(args.episodes):
        obs, _ = env.reset()
        state = obs_key(obs)
        while True:
            if rng.random() < epsilon:
                action = rng.randrange(len(ACTIONS))
            else:
                action = greedy_action(q_table, state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_key(next_obs)
            best_next = max(q_table[next_state])
            old_value = q_table[state][action]
            q_table[state][action] = old_value + args.lr * (reward + args.gamma * best_next - old_value)
            state = next_state

            if terminated or truncated:
                break

        epsilon = max(args.min_epsilon, epsilon * args.epsilon_decay)
        if args.report_every and (episode + 1) % args.report_every == 0:
            metrics = evaluate(q_table, episodes=50, seed=args.seed + episode + 1)
            print(
                f"episode={episode + 1} epsilon={epsilon:.3f} "
                f"success_rate={metrics['success_rate']:.2f} avg_reward={metrics['avg_reward']:.2f}"
            )

    return evaluate(q_table, episodes=args.eval_episodes, seed=args.seed + 999)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--oracle-epochs", type=int, default=25)
    parser.add_argument("--oracle-bonus", type=float, default=2.0)
    args = parser.parse_args()

    metrics = train(args)
    print(
        f"final success_rate={metrics['success_rate']:.2f} "
        f"avg_reward={metrics['avg_reward']:.2f} avg_length={metrics['avg_length']:.2f}"
    )


if __name__ == "__main__":
    main()
