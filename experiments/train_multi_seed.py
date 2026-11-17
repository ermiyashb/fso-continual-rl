"""
Multi-algorithm learning-curve comparison on FSOEnv (SAC / PPO / TD3 + baselines).

The logged paper figure was produced without --fast, with the defaults
--episodes 600 --seeds 3 (seeds 100, 101, 102). See METHODS_NOTES.md.

Usage (from the repository root):
    python experiments/train_multi_seed.py
    python experiments/train_multi_seed.py --plot-only
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from environment.fso_env import FSOEnv


class NormalizeObs(gym.ObservationWrapper):
    """Map the physical observation into [-1, 1] using the env bounds.

    Episode returns stay in the raw reward, because Monitor is underneath
    this wrapper.
    """

    def __init__(self, env):
        super().__init__(env)
        self._low = self.env.observation_space.low.astype(np.float32)
        self._high = self.env.observation_space.high.astype(np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones_like(self._low),
            high=np.ones_like(self._low),
            dtype=np.float32,
        )

    def observation(self, obs):
        scaled = 2.0 * (np.asarray(obs, dtype=np.float32) - self._low) / (self._high - self._low) - 1.0
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

MAX_STEPS = 80
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_FILE = RESULTS_DIR / "multi_algo_data.npz"
FIG_FILE = RESULTS_DIR / "multi_algo_comparison.png"


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
class EpisodeCollector(BaseCallback):
    """Collects the true per-episode return (from Monitor) and stops after n_episodes."""

    def __init__(self, n_episodes, tag):
        super().__init__()
        self.n_episodes = n_episodes
        self.tag = tag
        self.rewards = []
        self._last_pct = -1

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                self.rewards.append(float(ep["r"]))
        pct = int(100 * len(self.rewards) / self.n_episodes)
        if pct // 10 > self._last_pct // 10:
            self._last_pct = pct
            print(f"    [{self.tag}] {min(pct, 100)}%  ({len(self.rewards)} episodes)", flush=True)
        return len(self.rewards) < self.n_episodes


def make_model(name, env, seed, device="cpu", fast=False):
    common = dict(policy="MlpPolicy", env=env, seed=seed, verbose=0, device=device)
    # --fast: 4x fewer gradient updates for SAC/TD3 (big speedup, slightly less sample-efficient)
    off_policy = dict(train_freq=4, gradient_steps=1) if fast else {}
    if name == "PPO":
        # n_steps is a multiple of the 80-step episode. The Gaussian mean is
        # then placed at the center of the action box in train_one_run.
        return PPO(
            learning_rate=2.5e-4,
            n_steps=480,
            batch_size=96,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-2.0,
            ),
            **common,
        )
    if name == "SAC":
        return SAC(learning_rate=3e-4, learning_starts=500, batch_size=256, **off_policy, **common)
    if name == "TD3":
        n_act = env.action_space.shape[0]
        # TD3 is deterministic: without exploration noise it barely learns
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act))
        return TD3(learning_rate=3e-4, learning_starts=500, batch_size=256, action_noise=noise,
                   **off_policy, **common)
    raise ValueError(name)


def train_one_run(name, seed, n_episodes, tag, device="cpu", fast=False):
    # one thread per process: avoids CPU oversubscription when many runs go in parallel
    torch.set_num_threads(1)
    env = NormalizeObs(Monitor(FSOEnv(max_steps=MAX_STEPS)))
    model = make_model(name, env, seed, device=device, fast=fast)
    if name == "PPO":
        # PPO does not squash actions. A zero mean is clipped to the low
        # bound, so the policy starts at minimum power. Put the mean at the
        # center of each action and keep the noise small relative to that range.
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_((low + high) / 2.0)
            model.policy.log_std.copy_(torch.log(0.08 * (high - low)))
    cb = EpisodeCollector(n_episodes, tag)
    model.learn(total_timesteps=n_episodes * MAX_STEPS * 2, callback=cb, progress_bar=False)
    return np.array(cb.rewards[:n_episodes])


def _job(name, i, seed, n_episodes, tag, device, fast):
    return name, i, train_one_run(name, seed, n_episodes, tag, device, fast)


# ----------------------------------------------------------------------------
# Baselines (one value per episode, so they look like curves too)
# ----------------------------------------------------------------------------
def evaluate_policy(policy_fn, n_episodes, seed):
    env = FSOEnv(max_steps=MAX_STEPS)
    out = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs, env))
            total += r
            done = term or trunc
        out.append(total)
    return np.array(out)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def smooth(x, w):
    """Centered Gaussian smooth. Reflect padding keeps the curve the same length."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < 3:
        return x
    sigma = max(w / 4.0, 1.0)
    radius = int(w)
    t = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


STYLE = {
    "SAC": dict(color="#215CAF", ls="-", lw=2.8),
    "PPO": dict(color="#6C3483", ls="-", lw=2.8),
    "TD3": dict(color="#C0392B", ls="-", lw=2.8),
    "Fixed Policy": dict(color="#E67E22", ls="--", lw=2.4),
    "Random Policy": dict(color="#1E8449", ls=":", lw=2.4),
}


def make_plot(data, window, n_seeds):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    # Baselines first so the learning curves are drawn on top.
    series = [
        ("Random Policy", "random", 2),
        ("Fixed Policy", "fixed", 2),
        ("PPO", "PPO", 3),
        ("SAC", "SAC", 3),
        ("TD3", "TD3", 3),
    ]
    for name, key, z in series:
        if key not in data:
            continue
        values = data[key]
        if values.ndim == 2:
            curve = np.mean([smooth(run, window) for run in values], axis=0)
        else:
            curve = smooth(values, window)
        st = STYLE[name]
        ax.plot(
            np.arange(len(curve)),
            curve,
            label=name,
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            zorder=z,
            solid_capstyle="round",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward per Episode")
    ax.set_title(f"FSO Link Control  ·  mean over {n_seeds} seeds")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=1.0,
        edgecolor="#DDDDDD",
        borderpad=0.7,
    )
    ax.margins(x=0.01)
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_FILE}")
    plt.show()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--window", type=int, default=50, help="Gaussian smoothing width in episodes")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (0 = auto)")
    ap.add_argument("--device", default="cpu", help="cpu (recommended) or cuda")
    ap.add_argument("--fast", action="store_true", help="fewer SAC/TD3 gradient updates -> much faster")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        d = np.load(DATA_FILE)
        data = {k: d[k] for k in d.files}
        make_plot(data, args.window, data["SAC"].shape[0])
        return

    seeds = list(range(100, 100 + args.seeds))
    data = {}
    algos = ["SAC", "PPO", "TD3"]
    jobs = [(n, i, s) for n in algos for i, s in enumerate(seeds)]
    workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    print(f"Running {len(jobs)} training runs on {workers} parallel processes (device={args.device})")

    collected = {n: [None] * args.seeds for n in algos}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, n, i, s, args.episodes, f"{n} seed {i+1}/{args.seeds}",
                            args.device, args.fast) for n, i, s in jobs]
        for f in as_completed(futs):
            n, i, rewards = f.result()
            collected[n][i] = rewards
            print(f"  finished {n} seed {i+1}/{args.seeds}", flush=True)
    for n in algos:
        data[n] = np.array(collected[n])

    print("\n=== Baselines ===")
    fixed_action = np.array([10.0, 0.02], dtype=np.float32)
    data["fixed"] = evaluate_policy(lambda obs, env: fixed_action, args.episodes, seed=999)
    data["random"] = evaluate_policy(lambda obs, env: env.action_space.sample(), args.episodes, seed=888)
    print(f"Fixed mean:  {data['fixed'].mean():.2f}")
    print(f"Random mean: {data['random'].mean():.2f}")
    for n in ["SAC", "PPO", "TD3"]:
        print(f"{n} last-50-episode mean: {data[n][:, -50:].mean():.2f}")

    np.savez(DATA_FILE, **data)  # so you can re-plot without retraining
    make_plot(data, args.window, args.seeds)


if __name__ == "__main__":
    main()