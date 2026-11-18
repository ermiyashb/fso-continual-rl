"""
Held-out evaluation protocol and non-learning baselines for FSOEnv.

Every policy (learned or not) is scored with the same function,
`evaluate_policy`, on the same held-out environment seeds. Learned policies
act greedily (deterministic=True). Because the environment's random-number
consumption does not depend on the action taken, two policies evaluated on the
same env seed see exactly the same weather track and SNR noise sequence, so
per-seed comparisons are paired.

Baselines
---------
fixed       : always [10 dBm, 0.02 m].
random      : uniform sample from the action box (action space seeded).
rule_based  : if the current SNR < 10 dB or the current visibility < 3 km,
              use [15 dBm, 0.02 m]; otherwise use [10 dBm, 0.02 m].

The rule-based policy reads the raw (un-normalized) observation
[visibility_km, log10(Cn2), rx_power_dbm, snr_db]. The SNR entry is the SNR
measured on the previous step (the most recent SNR the transmitter can know);
the visibility entry is the current visibility. Actions are given in physical
units, which is exactly the env's action space ([4, 16] dBm x [0.01, 0.035] m),
so no rescaling is needed; they are clipped to the box for safety.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment.fso_env import FSOEnv

MAX_STEPS = 80

# Observation indices of FSOEnv
OBS_VIS, OBS_LOGCN2, OBS_RX, OBS_SNR = 0, 1, 2, 3

FIXED_ACTION = np.array([10.0, 0.02], dtype=np.float32)
RULE_SNR_DB = 10.0
RULE_VIS_KM = 3.0
RULE_HIGH_ACTION = np.array([15.0, 0.02], dtype=np.float32)
RULE_DEFAULT_ACTION = FIXED_ACTION

BASELINES = ("fixed", "random", "rule_based")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch global RNGs (env/action spaces are seeded separately)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def normalize_obs(obs, low, high):
    """Same affine map to [-1, 1] that NormalizeObs applies during training."""
    scaled = 2.0 * (np.asarray(obs, dtype=np.float32) - low) / (high - low) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Baseline policies. Signature: policy(raw_obs, env) -> action (physical units)
# ----------------------------------------------------------------------------
def fixed_policy(obs, env):
    return FIXED_ACTION.copy()


def random_policy(obs, env):
    return env.action_space.sample()


def rule_based_policy(obs, env):
    snr_db = float(obs[OBS_SNR])
    visibility_km = float(obs[OBS_VIS])
    if snr_db < RULE_SNR_DB or visibility_km < RULE_VIS_KM:
        action = RULE_HIGH_ACTION
    else:
        action = RULE_DEFAULT_ACTION
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


BASELINE_POLICIES = {
    "fixed": fixed_policy,
    "random": random_policy,
    "rule_based": rule_based_policy,
}


def model_policy(model):
    """Wrap a Stable-Baselines3 model as a greedy policy acting on raw observations."""

    def policy(obs, env):
        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)
        action, _ = model.predict(normalize_obs(obs, low, high), deterministic=True)
        return action

    return policy


# ----------------------------------------------------------------------------
# Evaluation protocol
# ----------------------------------------------------------------------------
def eval_env_seeds(seed_index: int, n_episodes: int, base: int) -> np.ndarray:
    """Held-out env seeds for one training seed index (disjoint across indices)."""
    start = base + seed_index * n_episodes
    return np.arange(start, start + n_episodes, dtype=np.int64)


def evaluate_policy(policy_fn, env_seeds, action_space_seed=None, max_steps=MAX_STEPS):
    """Run one episode per env seed.

    Returns a dict of per-episode arrays: return, mean_tx_dbm, mean_snr_db,
    outage_frac (fraction of steps with SNR < 6 dB, the env's outage flag).
    """
    env = FSOEnv(max_steps=max_steps)
    if action_space_seed is not None:
        env.action_space.seed(int(action_space_seed))
    out = {k: [] for k in ("return", "mean_tx_dbm", "mean_snr_db", "outage_frac")}
    for s in env_seeds:
        obs, _ = env.reset(seed=int(s))
        done, total, tx, snr, outage = False, 0.0, [], [], []
        while not done:
            action = np.clip(np.asarray(policy_fn(obs, env), dtype=np.float32),
                             env.action_space.low, env.action_space.high)
            obs, r, term, trunc, info = env.step(action)
            total += r
            tx.append(float(action[0]))
            snr.append(info["snr_db"])
            outage.append(info["outage"])
            done = term or trunc
        out["return"].append(total)
        out["mean_tx_dbm"].append(np.mean(tx))
        out["mean_snr_db"].append(np.mean(snr))
        out["outage_frac"].append(np.mean(outage))
    env.close()
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def evaluate_baseline(name, env_seeds, action_space_seed):
    return evaluate_policy(BASELINE_POLICIES[name], env_seeds, action_space_seed=action_space_seed)
