import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from environment.fso_env import FSOEnv


def evaluate_policy(env, policy_fn, n_episodes=30, seed=0):
    """Evaluate a policy and return list of episode rewards."""
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return np.array(rewards)


def main():
    env = FSOEnv(max_steps=100)

    # ----- 1. Fixed Policy (baseline) -----
    def fixed_policy(obs):
        # Always use medium power and medium beam
        return np.array([10.0, 0.02], dtype=np.float32)

    print("Evaluating Fixed Policy...")
    fixed_rewards = evaluate_policy(env, fixed_policy, n_episodes=40, seed=100)
    print(f"Fixed Policy - Mean: {fixed_rewards.mean():.2f} ± {fixed_rewards.std():.2f}")

    # ----- 2. Trained SAC -----
    print("Evaluating SAC...")
    model = SAC.load("results/sac_fso")

    def sac_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    sac_rewards = evaluate_policy(env, sac_policy, n_episodes=40, seed=100)
    print(f"SAC Policy     - Mean: {sac_rewards.mean():.2f} ± {sac_rewards.std():.2f}")

    # ----- Plot comparison -----
    plt.figure(figsize=(9, 5))

    # Box plot style comparison
    data = [fixed_rewards, sac_rewards]
    labels = ["Fixed Policy", "SAC"]

    bp = plt.boxplot(data, patch_artist=True,
                 boxprops=dict(facecolor="lightblue", alpha=0.7),
                 medianprops=dict(color="red", linewidth=2))
    plt.xticks(range(1, len(labels) + 1), labels)

    # Also show means
    means = [fixed_rewards.mean(), sac_rewards.mean()]
    plt.scatter([1, 2], means, color="darkred", zorder=5, label="Mean", s=60)

    plt.ylabel("Episode Reward")
    plt.title("Comparison: Fixed Policy vs SAC")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_fixed_vs_sac.png", dpi=150)
    print("Comparison plot saved to results/comparison_fixed_vs_sac.png")
    plt.show()


if __name__ == "__main__":
    main()