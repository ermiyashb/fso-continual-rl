"""(A) statistics and summary tables/figures for the revision.

Inputs : results/multi_algo_strong.npz          (original protocol, unchanged)
         results/revision/nonlearned.npz        (baselines_nonlearned.py)
         results/revision/runs/*.npz            (train_revision.py)
Outputs: results/revision/summary.json, results/revision/stats_*.csv,
         results/revision/tables/*.tex, results/revision/figures/*.png

Statistics are computed on the 10 per-seed mean held-out returns (one value per
seed index = mean of its 50 paired episodes). Primary tests: paired t-test
(scipy.stats.ttest_rel) and Wilcoxon signed-rank test (scipy.stats.wilcoxon,
exact two-sided), Holm-corrected within each comparison family. Welch tests are
reported for the original family only (appendix). IQM = 25% trimmed mean of
the 10 per-seed means; 95% CIs by percentile bootstrap over seeds (50,000
resamples; with a single task, rliable's stratified bootstrap reduces to this).
Usage: python experiments/revision/analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import stats

from common import REPO, N_SEEDS

RES = REPO / "results" / "revision"
TAB = RES / "tables"
FIG = RES / "figures"
B = 50_000


# ----------------------------------------------------------------------------
def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def boot_ci(x, fn, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    xs = x[idx]
    if fn is np.mean:
        vals = xs.mean(axis=1)
    else:   # IQM, vectorised exactly like scipy.stats.trim_mean(., 0.25)
        n = x.size
        k = int(0.25 * n)
        vals = np.sort(xs, axis=1)[:, k:n - k].mean(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def describe(per_seed):
    x = np.asarray(per_seed, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                mean_ci=boot_ci(x, np.mean), iqm=iqm(x), iqm_ci=boot_ci(x, iqm),
                per_seed=[round(float(v), 3) for v in x])


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_family(pairs, data):
    """pairs: list of (label, name_a, name_b); data[name] = per-seed means."""
    rows = []
    for lab, a, b in pairs:
        x, y = np.asarray(data[a]), np.asarray(data[b])
        d = x - y
        t = stats.ttest_rel(x, y)
        try:
            w = stats.wilcoxon(x, y, alternative="two-sided")
            wstat, wp = float(w.statistic), float(w.pvalue)
        except ValueError:
            wstat, wp = float("nan"), 1.0
        rows.append(dict(comparison=lab, a=a, b=b, diff=float(d.mean()),
                         diff_ci=boot_ci(d, np.mean), t=float(t.statistic), p_t=float(t.pvalue),
                         W=wstat, p_w=wp, higher=int((d > 0).sum())))
    for key in ("p_t", "p_w"):
        adj = holm([r[key] for r in rows])
        for r, v in zip(rows, adj):
            r[key + "_holm"] = float(v)
    return rows


def welch_family(pairs, data):
    rows = []
    for lab, a, b in pairs:
        x, y = np.asarray(data[a]), np.asarray(data[b])
        t = stats.ttest_ind(x, y, equal_var=False)
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        rows.append(dict(comparison=lab, diff=float(x.mean() - y.mean()), t=float(t.statistic),
                         df=float(df), p=float(t.pvalue), higher=int((x > y).sum())))
    adj = holm([r["p"] for r in rows])
    for r, v in zip(rows, adj):
        r["p_holm"] = float(v)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------
def load_all():
    orig = np.load(REPO / "results" / "multi_algo_strong.npz")
    nl = np.load(RES / "nonlearned.npz")
    pol = {}   # name -> dict(metric -> (10, 50) array)

    def add(name, getter):
        pol[name] = {m: getter(m) for m in ("return", "outage_frac", "mean_tx_dbm", "mean_w0_m", "mean_snr_db")}

    omap = {"return": "", "outage_frac": "_outage", "mean_tx_dbm": "_tx", "mean_snr_db": "_snr"}
    for p in ("TD3", "SAC", "PPO", "fixed", "random", "rule_based"):
        add(f"orig_{p}", lambda m, p=p: orig[f"eval_{p}{omap[m]}"] if m in omap else None)
    names = sorted({k.split("__")[0] for k in nl.files})
    for n in names:
        add(n, lambda m, n=n: nl[f"{n}__{m}"])
    runs = {}
    for f in sorted((RES / "runs").glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        d = np.load(f, allow_pickle=True)
        runs.setdefault((str(d["variant"]), str(d["algo"])), {})[int(d["seed_index"])] = d
    complete = {}
    for (v, a), per in runs.items():
        complete[(v, a)] = len(per)
        if len(per) < N_SEEDS:
            continue
        for tag in ("final", "best"):
            add(f"{v}_{a}_{tag}", lambda m, per=per, tag=tag: np.stack([per[i][f"{tag}_{m}"] for i in range(N_SEEDS)]))
    return orig, pol, runs, complete


def per_seed(pol, name, metric="return"):
    return pol[name][metric].mean(axis=1)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    orig, pol, runs, complete = load_all()
    ret = {n: per_seed(pol, n) for n in pol}
    summary = dict(runs_complete={f"{v}/{a}": c for (v, a), c in complete.items()}, policies={})
    for n, d in pol.items():
        s = describe(ret[n])
        s["outage"] = float(d["outage_frac"].mean())
        s["outage_seed_max"] = float(d["outage_frac"].mean(axis=1).max())
        s["mean_tx_dbm"] = float(d["mean_tx_dbm"].mean())
        s["mean_snr_db"] = float(d["mean_snr_db"].mean())
        s["mean_w0_cm"] = None if d["mean_w0_m"] is None else float(100 * d["mean_w0_m"].mean())
        s["p05_episode_return"] = float(np.percentile(d["return"], 5))
        summary["policies"][n] = s

    # ---- run-level extras (validation selection) ----
    sel = {}
    for (v, a), per in runs.items():
        if len(per) < N_SEEDS:
            continue
        sel[f"{v}_{a}"] = dict(best_val_episode=[int(per[i]["best_val_episode"]) for i in range(N_SEEDS)],
                               train_last50=float(np.mean([per[i]["train_returns"][-50:].mean() for i in range(N_SEEDS)])),
                               train_first50=float(np.mean([per[i]["train_returns"][:50].mean() for i in range(N_SEEDS)])),
                               wall_seconds=float(np.sum([per[i]["wall_seconds"] for i in range(N_SEEDS)])))
    summary["selection"] = sel

    # ---- families ----
    fam = {}
    A = ("TD3", "SAC", "PPO")
    pairs = []
    for a in A:
        for b, lb in (("fixed", "Fixed"), ("random", "Random"), ("rule_based", "Rule-based")):
            pairs.append((f"{a} vs {lb}", f"orig_{a}", f"orig_{b}"))
    pairs += [("SAC vs TD3", "orig_SAC", "orig_TD3"), ("SAC vs PPO", "orig_SAC", "orig_PPO"),
              ("TD3 vs PPO", "orig_TD3", "orig_PPO")]
    fam["original"] = paired_family(pairs, ret)
    fam["original_welch"] = welch_family(pairs, ret)

    if all(f"full_{a}_best" in ret for a in A):
        for tag in ("best", "final"):
            pr = []
            for a in A:
                for b, lb in (("fixed_grid_best", "best fixed"), ("rule_w0.02", "rule-based"),
                              ("rule_w0.035", "rule-based ($w_0$=3.5 cm)"), ("oracle_rewardmax", "oracle")):
                    pr.append((f"{a} vs {lb}", f"full_{a}_{tag}", b))
            fam[f"corrected_{tag}"] = paired_family(pr, ret)
        fam["best_vs_final"] = paired_family([(f"{a}: best-val vs final", f"full_{a}_best", f"full_{a}_final") for a in A], ret)
        fam["corrected_vs_original"] = paired_family(
            [(f"{a}: corrected best-val vs original", f"full_{a}_best", f"orig_{a}") for a in A] +
            [(f"{a}: corrected final vs original", f"full_{a}_final", f"orig_{a}") for a in A], ret)
    # partial observability: one Holm family per delay k (12 comparisons each)
    for v in ("po_k0", "po_k1", "po_k3"):
        pr = []
        for a in A + ("RPPO",):
            if f"{v}_{a}_best" not in ret:
                continue
            an = a.replace("RPPO", "Rec. PPO")
            pr.append((f"k={v[-1]}: {an} vs SNR rule", f"{v}_{a}_best", f"{v}_snr_rule_w0.035"))
            pr.append((f"k={v[-1]}: {an} vs gain tracking", f"{v}_{a}_best", f"{v}_gain_tracking"))
            pr.append((f"k={v[-1]}: {an} vs best fixed", f"{v}_{a}_best", "fixed_grid_best"))
        if pr:
            fam[f"po_best_{v[-2:]}"] = paired_family(pr, ret)
    summary["families"] = fam
    for k, rows in fam.items():
        write_csv(rows, RES / f"stats_{k}.csv")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {kk: (round(vv["mean"], 2), round(vv["iqm"], 2), round(vv["outage"], 4),
                                round(vv["mean_tx_dbm"], 2), vv["mean_w0_cm"] and round(vv["mean_w0_cm"], 2))
                           for kk, vv in [(k, summary["policies"][k])]} for k in summary["policies"]}, indent=0))
    for k, rows in fam.items():
        print("==", k)
        for r in rows:
            print("  ", {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk not in ("a", "b")})
    print(json.dumps(sel, indent=0))


if __name__ == "__main__":
    main()
