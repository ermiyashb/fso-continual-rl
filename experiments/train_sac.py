import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.fso_env import FSOEnv


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals.get("rewards", [])) > 0:
            self.current_rewards.append(self.locals["rewards"][0])

        if self.locals.get("dones", [False])[0]:
            if self.current_rewards:
                self.episode_rewards.append(sum(self.current_rewards))
                self.current_rewards = []
        return True


def main():
    env = Monitor(FSOEnv(max_steps=100))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=30_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=1,
        seed=42,
    )

    logger = RewardLogger()

    print("Training started...")
    model.learn(total_timesteps=15_000, callback=logger, progress_bar=True)
    print("Training finished.")

    model.save("results/sac_fso")

    if logger.episode_rewards:
        rewards = np.array(logger.episode_rewards)
        window = 8
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        else:
            moving_avg = rewards

        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.4, label="Episode Reward")
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f"Moving Avg ({window})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training – Reward vs Episode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/reward_curve_sac.png", dpi=150)
        print("Reward curve saved to results/reward_curve_sac.png")
        plt.show()


if __name__ == "__main__":
    main()