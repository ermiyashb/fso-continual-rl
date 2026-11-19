"""Shared pieces for the revision experiments.

Nothing in the original repository is modified. These helpers import the
original FSOEnv / physics and only wrap them:

* TruncateAtLimit   - reports the 80-step episode limit as truncation
                      (time-out) instead of termination.
* FullObsTime       - original 4-d observation normalised to [-1, 1] (same map as
                      experiments/train_multi_seed.NormalizeObs) plus a time feature.
* PartialObsDelay   - partially observable variant: no visibility, no Cn2. The
                      agent sees the last H SNR measurements delayed by k steps,
                      the last H actions it applied, and a time feature.

Evaluation (`run_episodes`) uses the same held-out env seeds and the same
metrics (return, mean P_tx, mean SNR, outage fraction = SNR < 6 dB) as
experiments/evaluation.py, and additionally records mean w0.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.fso_env import FSOEnv
from physics.fog_model import fog_transmission
from physics.fso_channel import gaussian_beam_radius, geometric_loss, dbm_to_watt

MAX_STEPS = 80
N_SEEDS = 10
TRAIN_SEEDS = list(range(100, 110))          # same as the paper
HELDOUT_BASE = 1_000_000                     # same as the paper (eval_seed_base)
HELDOUT_EPISODES = 50                        # per training-seed index
VAL_BASE = 2_000_000                         # validation env seeds, disjoint from held-out
VAL_EPISODES_CKPT = 20                       # episodes per checkpoint-selection evaluation
VAL_BASE_BASELINE = 3_000_000                # validation episodes for tuning non-learned baselines
VAL_EPISODES_BASELINE = 200

ACT_LOW = np.array([4.0, 0.01], dtype=np.float32)
ACT_HIGH = np.array([16.0, 0.035], dtype=np.float32)
SNR_SAT_DB = 12.0          # quality = clip((snr-1)/11, 0, 1) saturates at 12 dB
OUTAGE_DB = 6.0
ETA_TX = 0.8               # simulate_fso_channel defaults used by FSOEnv
ETA_RX = 0.8
RX_RADIUS = 0.025


def heldout_seeds(i: int) -> np.ndarray:
    s = HELDOUT_BASE + i * HELDOUT_EPISODES
    return np.arange(s, s + HELDOUT_EPISODES, dtype=np.int64)


def val_seeds_ckpt() -> np.ndarray:
    return np.arange(VAL_BASE, VAL_BASE + VAL_EPISODES_CKPT, dtype=np.int64)


def val_seeds_baseline() -> np.ndarray:
    return np.arange(VAL_BASE_BASELINE, VAL_BASE_BASELINE + VAL_EPISODES_BASELINE, dtype=np.int64)


# ----------------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------------
class TruncateAtLimit(gym.Wrapper):
    """FSOEnv sets terminated=True at max_steps. The link does not end there, so
    report it as truncated (time-out) so that value targets bootstrap."""

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        u = self.env.unwrapped
        if term and u.current_step >= u.max_steps:
            term, trunc = False, True
        return obs, r, term, trunc, info


def _scale(x, low, high):
    return np.clip(2.0 * (np.asarray(x, np.float32) - low) / (high - low) - 1.0, -1.0, 1.0)


def time_feature(env) -> float:
    u = env.unwrapped
    return float(1.0 - 2.0 * u.current_step / u.max_steps)   # +1 at start, -1 at the limit


class FullObsTime(gym.ObservationWrapper):
    """Original observation, normalised exactly as in training, plus time feature."""

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped.observation_space
        self._low = base.low.astype(np.float32)
        self._high = base.high.astype(np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(base.shape[0] + 1,), dtype=np.float32)

    def observation(self, obs):
        return np.append(_scale(obs, self._low, self._high), time_feature(self)).astype(np.float32)


class PartialObsDelay(gym.Wrapper):
    """Observation = [SNR_{t-k}, ..., SNR_{t-k-H+1}, a_t, ..., a_{t-H+1}, time].

    SNR_t is the SNR measured on step t (the env's info['snr_db']); it reaches
    the transmitter k steps later. a_t are the actions actually applied.
    Visibility and Cn2 are NOT observed. Before enough history exists the
    buffers are filled with the env's initial probe ([10 dBm, 0.02 m]) and its SNR.
    """

    SNR_LO, SNR_HI = -30.0, 30.0

    def __init__(self, env, delay: int = 0, history: int = 6):
        super().__init__(env)
        self.k = int(delay)
        self.H = int(history)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3 * self.H + 1,), dtype=np.float32)
        self._snr = deque(maxlen=self.H + self.k)
        self._act = deque(maxlen=self.H)

    def _obs(self):
        snr = list(self._snr)[::-1][self.k:self.k + self.H]     # newest first, skip k
        act = list(self._act)[::-1]
        s = _scale(np.array(snr, np.float32), self.SNR_LO, self.SNR_HI)
        a = np.concatenate([_scale(x, ACT_LOW, ACT_HIGH) for x in act])
        return np.concatenate([s, a, [time_feature(self)]]).astype(np.float32)

    def delayed_snr(self) -> float:
        """Raw delayed SNR (dB) currently available to the transmitter."""
        return float(list(self._snr)[::-1][self.k])

    def action_for_delayed_snr(self) -> np.ndarray:
        """Action that produced delayed_snr() (index k into the action history)."""
        return np.asarray(list(self._act)[::-1][min(self.k, self.H - 1)], np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        probe = np.array([10.0, 0.02], np.float32)
        snr0 = float(self.env.unwrapped.last_snr_db)
        self._snr.clear(); self._act.clear()
        for _ in range(self.H + self.k):
            self._snr.append(snr0)
        for _ in range(self.H):
            self._act.append(probe)
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), ACT_LOW, ACT_HIGH)
        obs, r, term, trunc, info = self.env.step(a)
        self._snr.append(float(info["snr_db"]))
        self._act.append(a)
        return self._obs(), r, term, trunc, info


def make_env(variant: str, monitor: bool = False):
    """variant: 'full' | 'po_k0' | 'po_k1' | 'po_k3'."""
    env = FSOEnv(max_steps=MAX_STEPS)
    if monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
    env = TruncateAtLimit(env)
    if variant == "full":
        return FullObsTime(env)
    if variant.startswith("po_k"):
        return PartialObsDelay(env, delay=int(variant[4:]))
    raise ValueError(variant)


# ----------------------------------------------------------------------------
# Channel model as implemented in FSOEnv (noise-free), for the oracle
# ----------------------------------------------------------------------------
_WANDER_ENV = FSOEnv(max_steps=MAX_STEPS)


def wander_gain(cn2: float, w_rx: float) -> float:
    _WANDER_ENV.cn2 = float(cn2)            # FSOEnv._wander_gain reads self.cn2
    return _WANDER_ENV._wander_gain(float(w_rx))


def channel_gain_db(vis_km: float, cn2: float, w0: float) -> float:
    """SNR(dB) - P_tx(dBm) of FSOEnv before the 0.15 dB measurement noise.

    P_r = P_tx * eta_t * eta_r * h_fog(V) * h_geo(w0) * h_bw(Cn2, w0); h_turb := 1 (mean).
    """
    # P_tx = 0 dBm -> 1 mW. rx_dbm = 10log10(P_w)+30, snr = rx_dbm - noise_floor.
    L = FSOEnv.DISTANCE_M
    lam = FSOEnv.WAVELENGTH_M
    h_fog = fog_transmission(visibility=vis_km, distance=L / 1000, wavelength=lam * 1e9)
    w = gaussian_beam_radius(w0, L, lam)
    h_geo = geometric_loss(RX_RADIUS, w)
    pr_w = dbm_to_watt(0.0) * ETA_TX * ETA_RX * h_fog * h_geo * wander_gain(cn2, w)
    return float(10 * np.log10(pr_w) + 30.0 - FSOEnv.NOISE_FLOOR_DBM)


W0_GRID_FINE = np.linspace(0.01, 0.035, 251)


def best_w0(vis_km: float, cn2: float) -> tuple[float, float]:
    """w0 in the action box that maximises the channel gain, and that gain (dB)."""
    g = np.array([channel_gain_db(vis_km, cn2, w) for w in W0_GRID_FINE])
    j = int(np.argmax(g))
    return float(W0_GRID_FINE[j]), float(g[j])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_episodes(variant, policy_fn, env_seeds, action_space_seed=None, trace=False):
    """policy_fn(obs, env) -> action in physical units. env is the wrapped env."""
    env = make_env(variant)
    if action_space_seed is not None:
        env.action_space.seed(int(action_space_seed))
    keys = ("return", "mean_tx_dbm", "mean_snr_db", "outage_frac", "mean_w0_m")
    out = {k: [] for k in keys}
    traces = []
    for s in env_seeds:
        obs, _ = env.reset(seed=int(s))
        done, total = False, 0.0
        tx, w0, snr, outage, vis, lc = [], [], [], [], [], []
        while not done:
            u = env.unwrapped
            vis.append(u.visibility); lc.append(np.log10(u.cn2))
            a = np.clip(np.asarray(policy_fn(obs, env), np.float32), ACT_LOW, ACT_HIGH)
            obs, r, term, trunc, info = env.step(a)
            total += r
            tx.append(float(a[0])); w0.append(float(a[1]))
            snr.append(info["snr_db"]); outage.append(info["outage"])
            done = term or trunc
        out["return"].append(total)
        out["mean_tx_dbm"].append(np.mean(tx))
        out["mean_snr_db"].append(np.mean(snr))
        out["outage_frac"].append(np.mean(outage))
        out["mean_w0_m"].append(np.mean(w0))
        if trace:
            traces.append(dict(tx=np.array(tx), w0=np.array(w0), snr=np.array(snr),
                               vis=np.array(vis), logcn2=np.array(lc)))
    env.close()
    res = {k: np.asarray(v, np.float64) for k, v in out.items()}
    if trace:
        res["traces"] = traces
    return res


def eval_heldout(variant, policy_factory, action_space_seed_fn=None):
    """Evaluate on all 10 x 50 held-out episodes. policy_factory(i) -> policy_fn."""
    per = []
    for i in range(N_SEEDS):
        ass = action_space_seed_fn(i) if action_space_seed_fn else None
        per.append(run_episodes(variant, policy_factory(i), heldout_seeds(i), ass))
    return {k: np.stack([p[k] for p in per]) for k in per[0]}


def raw_obs(env):
    """The original FSOEnv observation [V, log10 Cn2, rx_dbm, snr_db] (what the rule sees)."""
    return env.unwrapped._get_obs()


def model_policy(model):
    def pol(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return pol
