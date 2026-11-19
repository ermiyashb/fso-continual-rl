"""(C) corrected protocol and (D) partially observable variants: training + held-out eval.

Each run: one algorithm, one training seed (100..109), 1000 episodes x 80 steps,
hyperparameters from experiments/train_multi_seed.make_model (unchanged).
Differences from the original protocol, all implemented as wrappers:
  * the 80-step limit is a truncation (time-out), not a termination;
  * a time feature is appended to the observation;
  * every 50 training episodes the greedy policy is scored on 20 validation
    episodes (env seeds 2,000,000..2,000,019, disjoint from the held-out set);
    the best-validation checkpoint and the final checkpoint are both kept;
  * both are evaluated on the paper's held-out episodes for that seed index.

Variants: full (original observation + time), po_k0 / po_k1 / po_k3 (no V, no Cn2;
last 6 SNRs delayed by k steps, last 6 actions, time).

Usage (repo root):
  python experiments/revision/train_revision.py --variants full --algos TD3 SAC PPO --workers 6
  python experiments/revision/train_revision.py --variants po_k0 po_k1 po_k3 --algos SAC TD3 PPO --workers 6
Finished runs are skipped on restart (one npz per run in results/revision/runs/).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (MAX_STEPS, REPO, TRAIN_SEEDS, heldout_seeds, make_env, run_episodes,
                    val_seeds_ckpt)
from common import model_policy as _mlp_policy


def model_policy(model):
    """Greedy policy; for RecurrentPPO the LSTM state is reset at each episode start."""
    if type(model).__name__ != "RecurrentPPO":
        return _mlp_policy(model)
    st = {"h": None}

    def pol(obs, env):
        start = env.unwrapped.current_step == 0
        if start:
            st["h"] = None
        a, st["h"] = model.predict(obs, state=st["h"], episode_start=np.array([start]), deterministic=True)
        return a
    return pol

RUNS = REPO / "results" / "revision" / "runs"
MODELS = REPO / "results" / "revision" / "models"
N_EPISODES = 1000
VAL_EVERY = 50


def run_one(algo: str, variant: str, idx: int, n_episodes: int = N_EPISODES):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from evaluation import seed_everything
    from train_multi_seed import make_model

    out = RUNS / f"{variant}_{algo}_seed{TRAIN_SEEDS[idx]}.npz"
    if out.exists():
        return str(out), "skipped", 0.0
    t0 = time.time()
    seed = TRAIN_SEEDS[idx]
    torch.set_num_threads(1)
    seed_everything(seed)
    set_random_seed(seed)
    env = make_env(variant, monitor=True)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if algo == "RPPO":
        from sb3_contrib import RecurrentPPO
        # Same PPO hyperparameters as train_multi_seed.make_model, LSTM policy (sb3-contrib).
        model = RecurrentPPO("MlpLstmPolicy", env, learning_rate=2.5e-4, n_steps=480, batch_size=96,
                             n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.15, ent_coef=0.0,
                             vf_coef=0.5, max_grad_norm=0.5, seed=seed, verbose=0, device="cpu",
                             policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                                lstm_hidden_size=64, log_std_init=-2.0))
    else:
        model = make_model(algo, env, seed)
    if algo in ("PPO", "RPPO"):   # same initialisation as train_multi_seed.train_one_run
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_((low + high) / 2.0)
            model.policy.log_std.copy_(torch.log(0.08 * (high - low)))

    vseeds = val_seeds_ckpt()

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rewards, self.val_ep, self.val_ret = [], [], []
            self.best, self.best_ep, self.best_state = -np.inf, -1, None

        def _validate(self):
            r = run_episodes(variant, model_policy(self.model), vseeds)["return"].mean()
            ep = len(self.rewards)
            self.val_ep.append(ep); self.val_ret.append(float(r))
            if r > self.best:
                self.best, self.best_ep = float(r), ep
                self.best_state = copy.deepcopy(self.model.policy.state_dict())

        def _on_step(self):
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep is not None:
                    self.rewards.append(float(ep["r"]))
                    if len(self.rewards) % VAL_EVERY == 0 and len(self.rewards) <= n_episodes:
                        self._validate()
            return len(self.rewards) < n_episodes

    cb = CB()
    model.learn(total_timesteps=n_episodes * MAX_STEPS * 2, callback=cb, progress_bar=False)
    if len(cb.rewards) < n_episodes:
        raise RuntimeError(f"only {len(cb.rewards)} episodes")

    hs = heldout_seeds(idx)
    final = run_episodes(variant, model_policy(model), hs)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODELS / f"{variant}_{algo}_seed{seed}_final")
    model.policy.load_state_dict(cb.best_state)
    best = run_episodes(variant, model_policy(model), hs)
    model.save(MODELS / f"{variant}_{algo}_seed{seed}_bestval")

    data = dict(algo=algo, variant=variant, seed=seed, seed_index=idx,
                train_returns=np.array(cb.rewards[:n_episodes]),
                val_episodes=np.array(cb.val_ep), val_returns=np.array(cb.val_ret),
                best_val_episode=cb.best_ep, best_val_return=cb.best,
                heldout_env_seeds=hs, wall_seconds=time.time() - t0)
    for tag, r in (("final", final), ("best", best)):
        for k, v in r.items():
            data[f"{tag}_{k}"] = v
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    tmp.rename(out)
    return str(out), f"final {final['return'].mean():.2f} best {best['return'].mean():.2f} (ep {cb.best_ep})", time.time() - t0


def _safe(*a):
    try:
        return run_one(*a)
    except Exception:
        return f"{a}", "FAILED\n" + traceback.format_exc(), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["full"])
    ap.add_argument("--algos", nargs="+", default=["TD3", "SAC", "PPO"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--seed-idx", type=int, nargs="+", default=None,
                    help="optional subset of seed indices (0..9) to run; only selects jobs, "
                         "training is unchanged (used to split runs across machines)")
    args = ap.parse_args()
    idxs = args.seed_idx if args.seed_idx is not None else list(range(args.seeds))
    jobs = [(a, v, i, args.episodes) for v in args.variants for i in idxs for a in args.algos]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_safe, *j) for j in jobs]
        for f in as_completed(futs):
            name, msg, dt = f.result()
            print(f"[{time.time() - t0:7.0f}s] {Path(name).name}: {msg} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
