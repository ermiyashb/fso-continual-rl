"""(E) Does a policy adapt its power to the weather?

For every held-out step we compute the power the reward-maximising oracle
would send, P*(V, Cn2) (w0 = 3.5 cm, which maximises the channel gain over the
whole admissible weather range; checked on a 25 x 25 grid), and report per
policy the Pearson correlation between the policy's P_tx and P* over all
40,000 held-out steps, the RMS deviation |P_tx - P*|, the mean w0, and the
correlation of P_tx with visibility. A non-adaptive policy has correlation 0.
Usage: python experiments/revision/adaptation_metrics.py
Writes results/revision/adaptation_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import REPO, N_SEEDS, TRAIN_SEEDS, channel_gain_db, heldout_seeds, run_episodes
import policies as P

RES = REPO / "results" / "revision"
_PGRID = np.linspace(4.0, 16.0, 1201)


def p_star(vis, logc):
    g = channel_gain_db(float(vis), float(10 ** logc), 0.035)
    return float(_PGRID[int(np.argmax(P.expected_step_reward(_PGRID, g)))])


def collect(variant, factory):
    tx, w0, vis, ps = [], [], [], []
    for i in range(N_SEEDS):
        r = run_episodes(variant, factory(i), heldout_seeds(i), trace=True)
        for d in r["traces"]:
            tx.append(d["tx"]); w0.append(d["w0"]); vis.append(d["vis"])
            ps.append([p_star(v, c) for v, c in zip(d["vis"], d["logcn2"])])
    tx, w0, vis, ps = map(np.concatenate, (tx, w0, vis, ps))
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else 0.0
    return dict(corr_P_Pstar=corr(tx, ps), rms_P_minus_Pstar_db=float(np.sqrt(np.mean((tx - ps) ** 2))),
                corr_P_visibility=corr(tx, vis), mean_P=float(tx.mean()), sd_P=float(tx.std()),
                mean_w0_cm=float(100 * w0.mean()), sd_w0_cm=float(100 * w0.std()),
                frac_w0_ge_3cm=float((w0 >= 0.03).mean()))


def _job(name):
    import torch
    torch.set_num_threads(1)
    if name == "rule_based":
        return name, collect("full", lambda i: P.rule_policy(0.02))
    if name == "rule_w0.035":
        return name, collect("full", lambda i: P.rule_policy(0.035))
    if name == "fixed_grid_best":
        return name, collect("full", lambda i: P.fixed_policy(13.5, 0.035))
    if name == "oracle_rewardmax":
        return name, collect("full", lambda i: P.reward_max_oracle)
    from stable_baselines3 import PPO, SAC, TD3
    from train_revision import model_policy
    _, a, tag = name.split("_")
    cls = dict(SAC=SAC, TD3=TD3, PPO=PPO)[a]
    models = [cls.load(RES / "models" / f"full_{a}_seed{s}_{tag}.zip", device="cpu") for s in TRAIN_SEEDS]
    return name, collect("full", lambda i: model_policy(models[i]))


def main():
    from concurrent.futures import ProcessPoolExecutor
    names = ["rule_based", "rule_w0.035", "fixed_grid_best", "oracle_rewardmax"] + \
        [f"full_{a}_{t}" for a in ("TD3", "SAC", "PPO") for t in ("bestval", "final")]
    out = {}
    with ProcessPoolExecutor(8) as pool:
        for name, r in pool.map(_job, names):
            out[name] = r
            print(name, r, flush=True)
    out = {n: out[n] for n in names}
    (RES / "adaptation_metrics.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:22s}", {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
