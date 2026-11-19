"""
Multi-seed TD3 / SAC / PPO comparison on FSOEnv with a held-out greedy evaluation.

Two presets:

  strong (default)
      10 training seeds (100..109), 1000 training episodes per run. After
      training, the final policy of every run is evaluated greedily
      (deterministic=True) for 50 episodes on held-out environment seeds that
      were never used for training. The Fixed, Random and Rule-based baselines
      are evaluated with exactly the same protocol on exactly the same env
      seeds. Writes results/multi_algo_strong.npz, a summary table, and the
      figures results/strong_learning_curves.png, results/strong_eval_bar.png,
      results/strong_eval_box.png.

  legacy
      Re-runs the protocol behind the committed 3-seed / 600-episode figure
      (seeds 100, 101, 102; training-episode returns only; fixed rollout on
      env seeds 999.., random rollout on env seeds 888..). Output goes to
      results/multi_algo_data_rerun.npz and results/multi_algo_comparison_rerun.png
      so the committed results/multi_algo_data.npz is never overwritten.

Usage (from the repository root):
    python experiments/train_multi_seed.py                         # strong preset, everything
    python experiments/train_multi_seed.py --episodes 600          # strong protocol, 600 episodes
    python experiments/train_multi_seed.py --preset legacy         # old 3-seed / 600-episode run
    python experiments/train_multi_seed.py --plot-only             # redraw strong figures from npz
    python experiments/train_multi_seed.py --plot-only --preset legacy-committed
                                                                    # redraw the committed old figure
See README.md and METHODS_NOTES.md.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import gymnasium as gym
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from environment.fso_env import FSOEnv
from evaluation import (
    BASELINES,
    MAX_STEPS,
    evaluate_baseline,
    evaluate_policy,
    eval_env_seeds,
    model_policy,
    seed_everything,
)


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


ALGOS = ["TD3", "SAC", "PPO"]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Legacy (committed) files - never written by this script.
LEGACY_DATA_FILE = RESULTS_DIR / "multi_algo_data.npz"
LEGACY_FIG_FILE = RESULTS_DIR / "multi_algo_comparison.png"

PRESETS = {
    "strong": dict(episodes=1000, seeds=10, eval_episodes=50),
    "legacy": dict(episodes=600, seeds=3, eval_episodes=0),
}


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
        # TD3 is deterministic: without exploration noise it barely learns.
        # SB3 adds this noise to the action rescaled to [-1, 1], so sigma=0.1
        # is 5% of the half-range of each coordinate (not 0.1 dBm / 0.1 m).
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act))
        return TD3(learning_rate=3e-4, learning_starts=500, batch_size=256, action_noise=noise,
                   **off_policy, **common)
    raise ValueError(name)


def train_one_run(name, seed, n_episodes, tag, device="cpu", fast=False):
    """Train one run. Returns (training returns, model)."""
    # one thread per process: avoids CPU oversubscription when many runs go in parallel
    torch.set_num_threads(1)
    # Seed python / numpy / torch explicitly (SB3 also does this from `seed`).
    seed_everything(seed)
    set_random_seed(seed)
    env = NormalizeObs(Monitor(FSOEnv(max_steps=MAX_STEPS)))
    env.reset(seed=seed)  # env RNG
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
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
    rewards = np.array(cb.rewards[:n_episodes])
    if len(rewards) < n_episodes:
        raise RuntimeError(f"{tag}: only {len(rewards)} of {n_episodes} episodes completed")
    return rewards, model


def _job(name, i, seed, cfg):
    """Worker: train, then (strong protocol) evaluate the final greedy policy on held-out seeds."""
    tag = f"{name} seed {seed} ({i + 1}/{cfg['seeds']})"
    rewards, model = train_one_run(name, seed, cfg["episodes"], tag, cfg["device"], cfg["fast"])
    ev = None
    if cfg["eval_episodes"] > 0:
        seed_everything(cfg["eval_seed_base"] + i)
        env_seeds = eval_env_seeds(i, cfg["eval_episodes"], cfg["eval_seed_base"])
        ev = evaluate_policy(model_policy(model), env_seeds)
    if cfg.get("save_models_dir"):
        d = Path(cfg["save_models_dir"])
        d.mkdir(parents=True, exist_ok=True)
        model.save(d / f"{name}_seed{seed}")
    return name, i, rewards, ev


def _baseline_job(name, i, cfg):
    env_seeds = eval_env_seeds(i, cfg["eval_episodes"], cfg["eval_seed_base"])
    # Random policy: one action-space seed per seed index, so its spread across
    # seeds is a genuine across-seed spread.
    return name, i, evaluate_baseline(name, env_seeds, action_space_seed=cfg["eval_seed_base"] + 7919 * (i + 1))


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def smooth(x, w):
    """Centered Gaussian smooth. Reflect padding keeps the curve the same length."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) < 3:
        return x
    sigma = max(w / 4.0, 1.0)
    radius = int(min(w, len(x) - 1))
    t = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


STYLE = {
    "SAC": dict(color="#215CAF", ls="-", lw=2.4),
    "PPO": dict(color="#6C3483", ls="-", lw=2.4),
    "TD3": dict(color="#C0392B", ls="-", lw=2.4),
    "fixed": dict(color="#E67E22", ls="--", lw=2.0),
    "random": dict(color="#1E8449", ls=":", lw=2.0),
    "rule_based": dict(color="#555555", ls="-.", lw=2.0),
}
LABEL = {
    "TD3": "TD3", "SAC": "SAC", "PPO": "PPO",
    "fixed": "Fixed [10 dBm, 0.02 m]", "random": "Random", "rule_based": "Rule-based",
}
TICK = dict(LABEL, fixed="Fixed\n[10 dBm, 0.02 m]")


def _style():
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14, "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.linewidth": 0.8,
    })


def _spread(per_seed, band):
    per_seed = np.asarray(per_seed, dtype=float)
    n = per_seed.shape[0]
    sd = per_seed.std(axis=0, ddof=1) if n > 1 else np.zeros(per_seed.shape[1:])
    return sd / np.sqrt(n) if band == "se" else sd


def _finish(fig, path, show):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Strong-protocol data, summary and figures
# ----------------------------------------------------------------------------
def summarize(data, last_n=50):
    """Rows of summary statistics. Spread is across training seeds (per-seed means)."""
    algos = [str(a) for a in data["algos"]]
    policies = algos + [str(b) for b in data["baselines"]]
    rows = []
    for p in policies:
        ev = data[f"eval_{p}"]                    # (n_seeds, eval_episodes)
        per_seed = ev.mean(axis=1)
        n = len(per_seed)
        row = dict(
            policy=p,
            n_seeds=n,
            eval_episodes_per_seed=ev.shape[1],
            eval_mean=float(per_seed.mean()),
            eval_std_across_seeds=float(per_seed.std(ddof=1)) if n > 1 else 0.0,
            eval_se_across_seeds=float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            eval_std_all_episodes=float(ev.std(ddof=1)) if ev.size > 1 else 0.0,
            eval_mean_tx_dbm=float(data[f"eval_{p}_tx"].mean()),
            eval_mean_snr_db=float(data[f"eval_{p}_snr"].mean()),
            eval_outage_frac=float(data[f"eval_{p}_outage"].mean()),
            train_last_mean="", train_last_std_across_seeds="", train_all_mean="",
        )
        if p in algos:
            tr = data[f"train_{p}"]
            k = min(last_n, tr.shape[1])
            last = tr[:, -k:].mean(axis=1)
            row["train_last_mean"] = float(last.mean())
            row["train_last_std_across_seeds"] = float(last.std(ddof=1)) if n > 1 else 0.0
            row["train_all_mean"] = float(tr.mean())
        rows.append(row)
    return rows


def write_summary(rows, prefix, last_n):
    csv_path = Path(f"{prefix}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = [
        "| Policy | Held-out eval return (mean ± std across seeds) | SE | Outage frac | Mean P_tx (dBm) "
        f"| Training return, last {last_n} episodes (mean ± std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        tr = (f"{r['train_last_mean']:.2f} ± {r['train_last_std_across_seeds']:.2f}"
              if r["train_last_mean"] != "" else "n/a (not trained)")
        md.append(f"| {LABEL[r['policy']]} | {r['eval_mean']:.2f} ± {r['eval_std_across_seeds']:.2f} "
                  f"| {r['eval_se_across_seeds']:.2f} | {r['eval_outage_frac']:.3f} "
                  f"| {r['eval_mean_tx_dbm']:.2f} | {tr} |")
    md_path = Path(f"{prefix}_summary.md")
    md_path.write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"Saved: {csv_path}\nSaved: {md_path}")


def plot_strong(data, out_dir, window, band, show, tag="strong"):
    _style()
    algos = [str(a) for a in data["algos"]]
    baselines = [str(b) for b in data["baselines"]]
    n_seeds = int(data["n_seeds"])
    band_lbl = "± 1 SE" if band == "se" else "± 1 std"

    # 1) Training learning curves: smooth each seed, then mean ± spread across seeds.
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for b in baselines:
        m = data[f"eval_{b}"].mean()
        st = STYLE[b]
        ax.axhline(m, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=2,
                   label=f"{LABEL[b]} (held-out eval mean)")
    for a in algos:
        tr = data[f"train_{a}"]
        sm = np.array([smooth(run, window) for run in tr])
        mu, sp = sm.mean(axis=0), _spread(sm, band)
        x = np.arange(1, len(mu) + 1)
        st = STYLE[a]
        ax.plot(x, mu, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3, label=f"{a} (training)")
        ax.fill_between(x, mu - sp, mu + sp, color=st["color"], alpha=0.18, lw=0, zorder=1)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.set_title(f"Training returns, mean {band_lbl} over {n_seeds} seeds (Gaussian smooth {window})")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="#DDDDDD")
    _finish(fig, out_dir / f"{tag}_learning_curves.png", show)

    # 2) Held-out evaluation bar plot: mean of per-seed means, error = spread across seeds.
    policies = algos + baselines
    per_seed = [data[f"eval_{p}"].mean(axis=1) for p in policies]
    means = [v.mean() for v in per_seed]
    errs = [(_spread(v[:, None], band)[0]) for v in per_seed]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    xs = np.arange(len(policies))
    ax.bar(xs, means, yerr=errs, capsize=5, color=[STYLE[p]["color"] for p in policies],
           alpha=0.8, edgecolor="black", linewidth=0.6, zorder=2)
    rng = np.random.default_rng(0)  # jitter only
    for x, v in zip(xs, per_seed):
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color="black", alpha=0.7, zorder=3)
    ax.set_xticks(xs, [TICK[p] for p in policies])
    ax.set_ylabel("Held-out greedy evaluation return")
    lo = min(v.min() for v in per_seed)
    hi = max(v.max() for v in per_seed)
    pad = max(2.0, 0.1 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"Held-out evaluation ({int(data['eval_episodes'])} episodes/seed), "
                 f"mean {band_lbl} over {n_seeds} seeds")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    _finish(fig, out_dir / f"{tag}_eval_bar.png", show)

    # 3) Held-out evaluation box plot of per-seed mean returns.
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bp = ax.boxplot(per_seed, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    for patch, p in zip(bp["boxes"], policies):
        patch.set_facecolor(STYLE[p]["color"])
        patch.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(policies) + 1), [TICK[p] for p in policies])
    ax.set_ylabel("Per-seed mean held-out evaluation return")
    ax.set_title(f"Held-out evaluation, distribution over {n_seeds} seeds")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    _finish(fig, out_dir / f"{tag}_eval_box.png", show)


# ----------------------------------------------------------------------------
# Legacy figure (identical drawing to the committed multi_algo_comparison.png)
# ----------------------------------------------------------------------------
LEGACY_STYLE = {
    "SAC": dict(color="#215CAF", ls="-", lw=2.8),
    "PPO": dict(color="#6C3483", ls="-", lw=2.8),
    "TD3": dict(color="#C0392B", ls="-", lw=2.8),
    "Fixed Policy": dict(color="#E67E22", ls="--", lw=2.4),
    "Random Policy": dict(color="#1E8449", ls=":", lw=2.4),
}


def make_legacy_plot(data, window, n_seeds, fig_file, show):
    _style()
    plt.rcParams.update({"legend.fontsize": 11})
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    series = [("Random Policy", "random", 2), ("Fixed Policy", "fixed", 2),
              ("PPO", "PPO", 3), ("SAC", "SAC", 3), ("TD3", "TD3", 3)]
    for name, key, z in series:
        if key not in data:
            continue
        values = data[key]
        curve = (np.mean([smooth(run, window) for run in values], axis=0)
                 if values.ndim == 2 else smooth(values, window))
        st = LEGACY_STYLE[name]
        ax.plot(np.arange(len(curve)), curve, label=name, color=st["color"], ls=st["ls"],
                lw=st["lw"], zorder=z, solid_capstyle="round")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward per Episode")
    ax.set_title(f"FSO Link Control  ·  mean over {n_seeds} seeds")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=1.0,
              edgecolor="#DDDDDD", borderpad=0.7)
    ax.margins(x=0.01)
    _finish(fig, Path(fig_file), show)


def legacy_rollout(policy_name, n_episodes, seed):
    """Old baseline curve: one episode per env seed seed, seed+1, ... (action space seeded)."""
    from evaluation import BASELINE_POLICIES
    return evaluate_policy(BASELINE_POLICIES[policy_name], np.arange(seed, seed + n_episodes),
                           action_space_seed=seed)["return"]


# ----------------------------------------------------------------------------
def run_training(cfg, seeds, algos, workers):
    jobs = [(n, i, s) for n in algos for i, s in enumerate(seeds)]
    print(f"Running {len(jobs)} training runs on {workers} parallel processes (device={cfg['device']})")
    train = {n: [None] * len(seeds) for n in algos}
    evals = {n: [None] * len(seeds) for n in algos}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, n, i, s, cfg) for n, i, s in jobs]
        for f in as_completed(futs):
            n, i, rewards, ev = f.result()
            train[n][i], evals[n][i] = rewards, ev
            print(f"  finished {n} seed {seeds[i]}", flush=True)
    return train, evals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["strong", "legacy", "legacy-committed"], default="strong",
                    help="strong (default): 10 seeds x 1000 episodes + 50-episode held-out eval. "
                         "legacy: the old 3-seed / 600-episode training-return protocol. "
                         "legacy-committed: only valid with --plot-only, redraws the committed figure.")
    ap.add_argument("--episodes", type=int, default=None, help="training episodes per run (strong: 1000)")
    ap.add_argument("--seeds", type=int, default=None, help="number of training seeds (strong: 10)")
    ap.add_argument("--seed-base", type=int, default=100, help="training seeds are seed-base .. seed-base+seeds-1")
    ap.add_argument("--eval-episodes", type=int, default=None, help="held-out greedy eval episodes per seed (strong: 50)")
    ap.add_argument("--eval-seed-base", type=int, default=1_000_000,
                    help="first held-out env seed; seed index i uses base + i*eval_episodes + [0, eval_episodes)")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--window", type=int, default=50, help="Gaussian smoothing width in episodes (plots only)")
    ap.add_argument("--band", choices=["std", "se"], default="std", help="spread shown in plots")
    ap.add_argument("--last-n", type=int, default=50, help="training-return window reported in the summary")
    ap.add_argument("--out", default=None,
                    help="output prefix (strong default: results/multi_algo_strong)")
    ap.add_argument("--plot-only", action="store_true", help="redraw figures/summary from an existing npz")
    ap.add_argument("--save-models", action="store_true", help="save final models next to the npz, in models_<prefix name>/")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (0 = auto: about a third of CPU cores, to keep the machine usable)")
    ap.add_argument("--device", default="cpu", help="cpu (recommended) or cuda")
    ap.add_argument("--fast", action="store_true", help="fewer SAC/TD3 gradient updates -> much faster")
    ap.add_argument("--show", action="store_true", help="also open the figures in a window")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- legacy: committed figure only ----------------
    if args.preset == "legacy-committed":
        if not args.plot_only:
            ap.error("--preset legacy-committed is only valid with --plot-only")
        d = np.load(LEGACY_DATA_FILE)
        data = {k: d[k] for k in d.files}
        out = Path(args.out + ".png") if args.out else LEGACY_FIG_FILE
        make_legacy_plot(data, args.window, data["SAC"].shape[0], out, args.show)
        return

    preset = PRESETS[args.preset]
    episodes = args.episodes if args.episodes is not None else preset["episodes"]
    n_seeds = args.seeds if args.seeds is not None else preset["seeds"]
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else preset["eval_episodes"]
    default_prefix = "multi_algo_strong" if args.preset == "strong" else "multi_algo_data_rerun"
    prefix = Path(args.out) if args.out else RESULTS_DIR / default_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data_file = Path(str(prefix) + ".npz")
    fig_dir = prefix.parent
    fig_tag = "strong" if args.out is None else prefix.name

    # ---------------- legacy protocol ----------------
    if args.preset == "legacy":
        fig_file = RESULTS_DIR / "multi_algo_comparison_rerun.png" if args.out is None else Path(str(prefix) + ".png")
        if args.plot_only:
            d = np.load(data_file)
            data = {k: d[k] for k in d.files}
            make_legacy_plot(data, args.window, data["SAC"].shape[0], fig_file, args.show)
            return
        seeds = list(range(args.seed_base, args.seed_base + n_seeds))
        cfg = dict(seeds=n_seeds, episodes=episodes, device=args.device, fast=args.fast,
                   eval_episodes=eval_episodes, eval_seed_base=args.eval_seed_base, save_models_dir=None)
        workers = args.workers or max(1, min(len(seeds) * len(args.algos), max(1, (os.cpu_count() or 2) // 3)))
        train, _ = run_training(cfg, seeds, args.algos, workers)
        data = {n: np.array(train[n]) for n in args.algos}
        print("\n=== Baselines (legacy rollouts) ===")
        data["fixed"] = legacy_rollout("fixed", episodes, seed=999)
        data["random"] = legacy_rollout("random", episodes, seed=888)
        print(f"Fixed mean:  {data['fixed'].mean():.2f}")
        print(f"Random mean: {data['random'].mean():.2f}")
        for n in args.algos:
            print(f"{n} last-50-episode mean: {data[n][:, -50:].mean():.2f}")
        np.savez(data_file, **data)
        print(f"Saved: {data_file}")
        make_legacy_plot(data, args.window, n_seeds, fig_file, args.show)
        return

    # ---------------- strong protocol ----------------
    if args.plot_only:
        d = np.load(data_file, allow_pickle=False)
        data = {k: d[k] for k in d.files}
        write_summary(summarize(data, args.last_n), prefix, args.last_n)
        plot_strong(data, fig_dir, args.window, args.band, args.show, tag=fig_tag)
        return

    if eval_episodes <= 0:
        ap.error("the strong preset needs --eval-episodes > 0")
    seeds = list(range(args.seed_base, args.seed_base + n_seeds))
    models_dir = (prefix.parent / f"models_{prefix.name}") if args.save_models else None
    cfg = dict(seeds=n_seeds, episodes=episodes, device=args.device, fast=args.fast,
               eval_episodes=eval_episodes, eval_seed_base=args.eval_seed_base,
               save_models_dir=str(models_dir) if models_dir else None)
    workers = args.workers or max(1, min(len(seeds) * len(args.algos), max(1, (os.cpu_count() or 2) // 3)))
    seed_everything(args.seed_base)
    train, evals = run_training(cfg, seeds, args.algos, workers)

    print("\n=== Baselines (same held-out protocol) ===")
    base = {b: [None] * n_seeds for b in BASELINES}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_baseline_job, b, i, cfg) for b in BASELINES for i in range(n_seeds)]
        for f in as_completed(futs):
            b, i, ev = f.result()
            base[b][i] = ev

    data = dict(
        algos=np.array(args.algos),
        baselines=np.array(BASELINES),
        train_seeds=np.array(seeds),
        n_seeds=np.array(n_seeds),
        episodes=np.array(episodes),
        eval_episodes=np.array(eval_episodes),
        eval_seed_base=np.array(args.eval_seed_base),
        eval_env_seeds=np.stack([eval_env_seeds(i, eval_episodes, args.eval_seed_base) for i in range(n_seeds)]),
        max_steps=np.array(MAX_STEPS),
        fast=np.array(args.fast),
        config_json=np.array(json.dumps(dict(vars(args), episodes=episodes, seeds=n_seeds,
                                             eval_episodes=eval_episodes))),
    )
    for n in args.algos:
        data[f"train_{n}"] = np.array(train[n])
    sources = {n: evals[n] for n in args.algos}
    sources.update(base)
    for p, per_seed in sources.items():
        data[f"eval_{p}"] = np.stack([e["return"] for e in per_seed])
        data[f"eval_{p}_tx"] = np.stack([e["mean_tx_dbm"] for e in per_seed])
        data[f"eval_{p}_snr"] = np.stack([e["mean_snr_db"] for e in per_seed])
        data[f"eval_{p}_outage"] = np.stack([e["outage_frac"] for e in per_seed])

    np.savez(data_file, **data)
    print(f"Saved: {data_file}")
    write_summary(summarize(data, args.last_n), prefix, args.last_n)
    plot_strong(data, fig_dir, args.window, args.band, args.show, tag=fig_tag)


if __name__ == "__main__":
    main()
