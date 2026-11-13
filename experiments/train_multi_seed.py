import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.fso_env import FSOEnv


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, algo_name, seed_idx, n_seeds):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.algo_name = algo_name
        self.seed_idx = seed_idx
        self.n_seeds = n_seeds
        self.episode_rewards = []
        self.current_reward = 0.0
        self.last_print = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_reward += reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        progress = int(100 * self.num_timesteps / self.total_timesteps)
        if progress > self.last_print:
            self.last_print = progress
            print(f"    [{self.algo_name} {self.seed_idx}/{self.n_seeds}] {progress}% "
                  f"| ep={len(self.episode_rewards)}", flush=True)
        return True


def train_one_run(algo_class, algo_name, seed, seed_idx, n_seeds, total_timesteps=6000):
    env = Monitor(FSOEnv(max_steps=80))
    model = algo_class("MlpPolicy", env, learning_rate=3e-4, verbose=0, seed=seed)
    cb = ProgressCallback(total_timesteps, algo_name, seed_idx, n_seeds)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    return np.array(cb.episode_rewards)


def evaluate_fixed(n_episodes=100, seed=0):
    """Fixed policy: constant power and beam."""
    env = FSOEnv(max_steps=80)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_rew = 0.0
        while not done:
            action = np.array([10.0, 0.02], dtype=np.float32)
            obs, r, term, trunc, _ = env.step(action)
            ep_rew += r
            done = term or trunc
        rewards.append(ep_rew)
    return np.array(rewards)


def main():
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))
    total_timesteps = 6000

    results = {"SAC": [], "PPO": []}

    for name, algo in [("SAC", SAC), ("PPO", PPO)]:
        print(f"\n=== Training {name} ({n_seeds} seeds) ===")
        for i, seed in enumerate(seeds):
            print(f"  Seed {i+1}/{n_seeds}")
            rewards = train_one_run(algo, name, seed, i+1, n_seeds, total_timesteps)
            if len(rewards) > 0:
                results[name].append(rewards)
            print(f"  Done | episodes={len(rewards)}\n")

    # Fixed policy (no training needed)
    print("=== Evaluating Fixed Policy ===")
    fixed_rewards = evaluate_fixed(n_episodes=100, seed=999)
    print(f"Fixed Policy mean: {fixed_rewards.mean():.2f}")

    # Align lengths
    min_len = min(len(r) for runs in results.values() for r in runs)
    print(f"\nPlotting up to {min_len} episodes")

    plt.figure(figsize=(10, 6))
    colors = {"SAC": "C0", "PPO": "C1"}

    for name in ["SAC", "PPO"]:
        data = np.array([r[:min_len] for r in results[name]])
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ep = np.arange(min_len)
        plt.plot(ep, mean, label=name, color=colors[name], linewidth=2)
        plt.fill_between(ep, mean - std, mean + std, alpha=0.2, color=colors[name])

    # Fixed policy as horizontal line (mean ± std)
    fixed_mean = fixed_rewards.mean()
    fixed_std = fixed_rewards.std()
    plt.axhline(fixed_mean, color="C2", linestyle="--", linewidth=2, label="Fixed Policy")
    plt.fill_between([0, min_len], fixed_mean - fixed_std, fixed_mean + fixed_std,
                     alpha=0.15, color="C2")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Mean Reward ± Std ({n_seeds} seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/multi_seed_comparison.png", dpi=150)
    print("Saved: results/multi_seed_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()