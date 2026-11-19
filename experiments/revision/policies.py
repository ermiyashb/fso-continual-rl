"""Non-learned policies for the revision (all act in physical units)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from common import (ACT_HIGH, ACT_LOW, SNR_SAT_DB, best_w0, raw_obs)
import evaluation as E   # original experiments/evaluation.py (rule thresholds, fixed action)

NOISE_SD_DB = 0.15     # FSOEnv adds N(0, 0.15) dB to the measured SNR
POWER_COST_PER_DB = 0.16 / 12.0
QUAL_SLOPE_PER_DB = 1.0 / 11.0


def rule_policy(w0: float):
    """Original rule (SNR < 10 dB or V < 3 km -> 15 dBm else 10 dBm) with waist w0."""
    def pol(obs, env):
        o = raw_obs(env)
        hi = o[E.OBS_SNR] < E.RULE_SNR_DB or o[E.OBS_VIS] < E.RULE_VIS_KM
        return np.array([15.0 if hi else 10.0, w0], np.float32)
    return pol


def fixed_policy(p_dbm: float, w0: float):
    a = np.array([p_dbm, w0], np.float32)
    return lambda obs, env: a


def _vis_cn2(env):
    o = raw_obs(env)              # what the agent observes (float32, clipped)
    return float(o[E.OBS_VIS]), float(10 ** o[E.OBS_LOGCN2])


def min_power_oracle(obs, env):
    """Least P_tx that puts the noise-free SNR at the 12 dB saturation point,
    using the best w0 and the observed V and Cn2 (clipped to the action box)."""
    v, c = _vis_cn2(env)
    w, g = best_w0(v, c)
    return np.array([np.clip(SNR_SAT_DB - g, ACT_LOW[0], ACT_HIGH[0]), w], np.float32)


# Reward-maximising margin above saturation with Gaussian SNR noise:
# d/dm E[quality] = slope * P(noise < -m) = power cost  ->  m* = sd * Phi^{-1}(1 - cost/slope)
REWARD_MARGIN_DB = float(NOISE_SD_DB * norm.ppf(1.0 - POWER_COST_PER_DB / QUAL_SLOPE_PER_DB))


def expected_step_reward(p, g):
    """E[quality - cost] (without the 100/80 factor) for SNR = p + g + N(0, sd)."""
    mu = (p + g - 1.0) / 11.0
    s = NOISE_SD_DB / 11.0
    # E[clip(X,0,1)] for X ~ N(mu, s)
    def e_pos(m):  # E[max(X - m, 0)] shifted
        z = (mu - m) / s
        return (mu - m) * norm.cdf(z) + s * norm.pdf(z)
    q = e_pos(0.0) - e_pos(1.0)
    return q - POWER_COST_PER_DB * (p - 4.0)


_PGRID = np.linspace(4.0, 16.0, 1201)


def reward_max_oracle(obs, env):
    """Per-step expected-reward maximiser given observed V, Cn2 (weather is exogenous,
    so the per-step greedy maximiser is optimal for the episode)."""
    v, c = _vis_cn2(env)
    w, g = best_w0(v, c)
    p = _PGRID[int(np.argmax(expected_step_reward(_PGRID, g)))]
    return np.array([p, w], np.float32)


# ----------------------------------------------------------------------------
# Information-matched baselines for the partially observable variants
# (they only use what PartialObsDelay exposes: delayed SNR and own past actions)
# ----------------------------------------------------------------------------
def delayed_snr_rule(w0: float = 0.02):
    """SNR clause of the original rule on the delayed SNR (no visibility available)."""
    def pol(obs, env):
        hi = env.delayed_snr() < E.RULE_SNR_DB
        return np.array([15.0 if hi else 10.0, w0], np.float32)
    return pol


def gain_tracking(margin_db: float, w0: float = 0.035):
    """Certainty-equivalent controller: estimate the channel gain from the
    delayed SNR and the power that produced it, then send the power that would
    reach 12 dB + margin if the gain had not changed."""
    def pol(obs, env):
        a_prev = env.action_for_delayed_snr()
        g_hat = env.delayed_snr() - float(a_prev[0])
        p = np.clip(SNR_SAT_DB + margin_db - g_hat, ACT_LOW[0], ACT_HIGH[0])
        return np.array([p, w0], np.float32)
    return pol
