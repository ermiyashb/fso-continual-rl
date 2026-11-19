"""(B) and (D-baselines): non-learned policies on the paper's held-out episodes.

Held-out: env seeds 1,000,000 + 50*i + j, i = 0..9 (seed index), j = 0..49
(identical to experiments/train_multi_seed.py --preset strong).
Tuning (grid-searched fixed action, gain-tracking margin) uses separate
validation episodes, env seeds 3,000,000..3,000,199, never the held-out set.

Usage (repo root):  python experiments/revision/baselines_nonlearned.py --workers 6
Writes results/revision/nonlearned.npz and results/revision/nonlearned_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import REPO, eval_heldout, run_episodes, val_seeds_baseline
import policies as P

OUT = REPO / "results" / "revision"

P_GRID = np.round(np.arange(4.0, 16.0 + 1e-9, 0.5), 3)
W_GRID = np.round(np.arange(0.01, 0.035 + 1e-9, 0.0025), 4)
MARGINS = np.round(np.arange(0.0, 4.0 + 1e-9, 0.25), 3)


def _val_fixed(pw):
    p, w = pw
    r = run_episodes("full", P.fixed_policy(p, w), val_seeds_baseline())
    return p, w, float(r["return"].mean())


def _val_track(args):
    variant, m = args
    r = run_episodes(variant, P.gain_tracking(m), val_seeds_baseline())
    return variant, m, float(r["return"].mean())


def _held(args):
    name, variant, kind, params = args
    if kind == "fixed":
        f = lambda i: P.fixed_policy(*params)
    elif kind == "rule":
        f = lambda i: P.rule_policy(params)
    elif kind == "minpow":
        f = lambda i: P.min_power_oracle
    elif kind == "rewmax":
        f = lambda i: P.reward_max_oracle
    elif kind == "dsnr_rule":
        f = lambda i: P.delayed_snr_rule(params)
    elif kind == "track":
        f = lambda i: P.gain_tracking(params)
    else:
        raise ValueError(kind)
    return name, eval_heldout(variant, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sel = {}
    with ProcessPoolExecutor(args.workers) as pool:
        grid = [(p, w) for p in P_GRID for w in W_GRID]
        res = list(pool.map(_val_fixed, grid, chunksize=4))
        best = max(res, key=lambda t: t[2])
        sel["fixed_grid"] = dict(power_dbm=best[0], w0_m=best[1], val_return=best[2],
                                 grid_power=P_GRID.tolist(), grid_w0=W_GRID.tolist(),
                                 val_env_seeds=[int(val_seeds_baseline()[0]), int(val_seeds_baseline()[-1])],
                                 table=[list(map(float, t)) for t in res])
        print("best fixed (validation):", best, flush=True)
        tr = list(pool.map(_val_track, [(v, m) for v in ("po_k0", "po_k1", "po_k3") for m in MARGINS]))
        for v in ("po_k0", "po_k1", "po_k3"):
            rows = [t for t in tr if t[0] == v]
            b = max(rows, key=lambda t: t[2])
            sel[f"track_{v}"] = dict(margin_db=b[1], val_return=b[2], table=[[t[1], t[2]] for t in rows])
            print("tracking", v, b, flush=True)

        jobs = [
            ("fixed_10_0.02", "full", "fixed", (10.0, 0.02)),
            ("fixed_grid_best", "full", "fixed", (sel["fixed_grid"]["power_dbm"], sel["fixed_grid"]["w0_m"])),
            ("rule_w0.02", "full", "rule", 0.02),
            ("rule_w0.035", "full", "rule", 0.035),
            ("oracle_minpower", "full", "minpow", None),
            ("oracle_rewardmax", "full", "rewmax", None),
        ]
        for v in ("po_k0", "po_k1", "po_k3"):
            jobs += [
                (f"{v}_snr_rule_w0.02", v, "dsnr_rule", 0.02),
                (f"{v}_snr_rule_w0.035", v, "dsnr_rule", 0.035),
                (f"{v}_gain_tracking", v, "track", sel[f"track_{v}"]["margin_db"]),
            ]
        data = {}
        for name, r in pool.map(_held, jobs):
            for k, val in r.items():
                data[f"{name}__{k}"] = val
            print(f"{name:28s} return {r['return'].mean():6.2f}  outage {r['outage_frac'].mean():.4f}  "
                  f"P {r['mean_tx_dbm'].mean():5.2f}  w0 {r['mean_w0_m'].mean():.4f}", flush=True)
    sel["reward_margin_db_analytic"] = P.REWARD_MARGIN_DB
    np.savez(OUT / "nonlearned.npz", **data)
    (OUT / "nonlearned_selection.json").write_text(json.dumps(sel, indent=1))
    print("saved", OUT / "nonlearned.npz")


if __name__ == "__main__":
    main()
