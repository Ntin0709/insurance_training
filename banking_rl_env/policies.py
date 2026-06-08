"""Reference policies and evaluation helpers."""

from __future__ import annotations

from banking_rl_env.tools import ACTIONS


def action_index(action_name: str) -> int:
    return ACTIONS.index(action_name)


def oracle_action(env) -> int:
    """Return the next action in the current scenario's ideal trajectory."""
    for action_name in env.scenario.oracle_actions():
        if action_name == "AUTHENTICATE" and env.simulator.authenticated:
            continue
        if action_name == "ASK_CONFIRMATION" and env.simulator.confirmed:
            continue
        if action_name in env.simulator.called_tools:
            continue
        if action_name == "FINISH" and not env.target_called:
            continue
        return action_index(action_name)
    return action_index("FINISH")


def run_policy(env, policy, episodes: int = 20) -> dict[str, float]:
    total_reward = 0.0
    successes = 0
    lengths = 0
    for _ in range(episodes):
        env.reset()
        episode_reward = 0.0
        while True:
            _, reward, terminated, truncated, info = env.step(policy(env))
            episode_reward += reward
            if terminated or truncated:
                successes += int(info["last_event"] == "success")
                lengths += env.step_count
                break
        total_reward += episode_reward
    return {
        "episodes": float(episodes),
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_length": lengths / episodes,
    }
