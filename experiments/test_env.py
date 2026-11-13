import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from environment.fso_env import FSOEnv

env = FSOEnv()
obs, info = env.reset(seed=42)
print("Initial obs:", obs)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1} | Reward: {reward:.3f} | SNR: {info['snr_db']:.2f} dB | Visibility: {info['visibility']:.2f} km")

    if terminated:
        break