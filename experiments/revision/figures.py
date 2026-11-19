"""Figures for the revision (E and summary plots).
Usage: python experiments/revision/figures.py   (after analysis.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import REPO, N_SEEDS, heldout_seeds, run_episodes, val_seeds_ckpt, raw_obs
import policies as P
from train_multi_seed import STYLE, _style, smooth   # original plotting style
import matplotlib.pyplot as plt

RES = REPO / "results" / "revision"
FIG = RES / "figures"
COL = dict(STYLE)
COL.update({
    "grid": dict(color="#B9770E", ls="--", lw=2.0),
    "rule35": dict(color="#222222", ls="-.", lw=2.0),
    "oracle": dict(color="#000000", ls=":", lw=2.0),
    "track": dict(color="#117A65", ls="--", lw=2.0),
    "RPPO": dict(color="#AF7AC5", ls="-", lw=2.4),
})


def load_runs(variant, algo):
    out = []
    for i, s in enumerate(range(100, 110)):
        f = RES / "runs" / f"{variant}_{algo}_seed{s}.npz"
        if not f.exists():
            return None
        out.append(np.load(f, allow_pickle=True))
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / name)


def bars(ax, entries, summ):
    """entries: list of (label, policy key, color)."""
    for j, (lab, key, c) in enumerate(entries):
        s = summ[key]
        ax.bar(j, s["mean"], color=c, alpha=0.75, width=0.7, edgecolor="black", lw=0.6)
        ax.errorbar(j, s["mean"], yerr=s["sd"], color="black", capsize=3, lw=1)
        ps = np.array(s["per_seed"])
        ax.scatter(j + np.linspace(-0.2, 0.2, len(ps)), ps, s=9, color="black", zorder=3)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], rotation=40, ha="right", fontsize=9)


def fig_main(summ):
    e = []
    for tag, lab in (("orig", "original, final"), ("final", "corrected, final"), ("best", "corrected, best-val")):
        for a in ("TD3", "SAC", "PPO"):
            key = f"orig_{a}" if tag == "orig" else f"full_{a}_{tag}"
            e.append((f"{a} ({lab})", key, COL[a]["color"]))
    e += [("Fixed [10 dBm, 2 cm]", "fixed_10_0.02", COL["fixed"]["color"]),
          ("Best fixed [13.5 dBm, 3.5 cm]", "fixed_grid_best", COL["grid"]["color"]),
          ("Rule-based", "rule_w0.02", COL["rule_based"]["color"]),
          ("Rule-based ($w_0$ = 3.5 cm)", "rule_w0.035", COL["rule35"]["color"]),
          ("Oracle (reward-max.)", "oracle_rewardmax", "#BBBBBB")]
    _style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars(ax, e, summ)
    for x in (2.5, 5.5, 8.5):
        ax.axvline(x, color="grey", lw=0.6, ls=":")
    ax.set_ylabel("Held-out return")
    ax.set_ylim(30, 95)
    save(fig, "fig_rev_main.png")


def val_reference():
    """Non-learned references on the 20 checkpoint-selection validation episodes."""
    vs = val_seeds_ckpt()
    return {k: float(run_episodes("full", f, vs)["return"].mean()) for k, f in (
        ("Rule-based", P.rule_policy(0.02)), ("Best fixed", P.fixed_policy(13.5, 0.035)),
        ("Oracle", P.reward_max_oracle))}


def fig_curves(sel_json):
    _style()
    ref = val_reference()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for a in ("TD3", "SAC", "PPO"):
        runs = load_runs("full", a)
        tr = np.stack([r["train_returns"] for r in runs])
        m, s = smooth(tr.mean(0), 50), smooth(tr.std(0, ddof=1), 50)
        x = np.arange(1, tr.shape[1] + 1)
        axs[0].plot(x, m, label=a, **COL[a]); axs[0].fill_between(x, m - s, m + s, color=COL[a]["color"], alpha=0.15)
        ve = runs[0]["val_episodes"]
        vr = np.stack([r["val_returns"] for r in runs])
        axs[1].plot(ve, vr.mean(0), marker="o", ms=3, label=a, **COL[a])
        axs[1].fill_between(ve, vr.mean(0) - vr.std(0, ddof=1), vr.mean(0) + vr.std(0, ddof=1), color=COL[a]["color"], alpha=0.15)
    for k, st in (("Rule-based", COL["rule_based"]), ("Best fixed", COL["grid"]), ("Oracle", COL["oracle"])):
        axs[1].axhline(ref[k], label=f"{k} (validation)", **{**st, "lw": 1.4})
    axs[0].set_xlabel("Training episode"); axs[0].set_ylabel("Training return (smoothed, window 50)")
    axs[1].set_xlabel("Training episode"); axs[1].set_ylabel("Validation return (20 episodes)")
    axs[0].set_title("(a) training"); axs[1].set_title("(b) validation checkpoints")
    axs[0].legend(loc="lower left"); axs[1].legend(loc="lower left", fontsize=8)
    axs[0].set_ylim(30, 95); axs[1].set_ylim(30, 95)
    save(fig, "fig_rev_curves.png")
    return ref


def fig_po(summ):
    _style()
    ks = ("po_k0", "po_k1", "po_k3")
    algos = [a for a in ("TD3", "SAC", "PPO", "RPPO") if f"po_k0_{a}_best" in summ]
    items = [(a, f"{{}}_{a}_best", COL[a]["color"]) for a in algos] + [
        ("SNR rule (3.5 cm)", "{}_snr_rule_w0.035", COL["rule35"]["color"]),
        ("Gain tracking", "{}_gain_tracking", COL["track"]["color"])]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    W = 0.8 / len(items)
    for j, (lab, pat, c) in enumerate(items):
        for g, k in enumerate(ks):
            s = summ[pat.format(k)]
            x = g + (j - (len(items) - 1) / 2) * W
            ax.bar(x, s["mean"], width=W * 0.95, color=c, alpha=0.75, edgecolor="black", lw=0.5,
                   label=lab.replace("RPPO", "Recurrent PPO") + (" (best-val)" if lab in algos else "") if g == 0 else None)
            ax.errorbar(x, s["mean"], yerr=s["sd"], color="black", capsize=2, lw=0.8)
            ax.scatter(np.full(N_SEEDS, x), s["per_seed"], s=5, color="black", zorder=3)
    ax.axhline(summ["oracle_rewardmax"]["mean"], label="Full-information oracle", **COL["oracle"])
    ax.axhline(summ["fixed_grid_best"]["mean"], label="Best constant action", **COL["grid"])
    ax.set_xticks(range(3)); ax.set_xticklabels([f"feedback delay k = {k[-1]}" for k in ks])
    ax.set_ylabel("Held-out return"); ax.set_ylim(60, 93)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    save(fig, "fig_rev_po.png")


def fig_adapt():
    from stable_baselines3 import SAC
    from train_revision import model_policy
    seed_env = int(heldout_seeds(0)[0])         # first held-out episode of seed index 0 (fixed a priori)
    m_best = SAC.load(RES / "models" / "full_SAC_seed100_bestval.zip", device="cpu")
    m_fin = SAC.load(RES / "models" / "full_SAC_seed100_final.zip", device="cpu")
    pols = [("SAC best-val (seed 100)", model_policy(m_best), COL["SAC"]),
            ("SAC final (seed 100)", model_policy(m_fin), dict(color="#7FB3D5", ls="--", lw=1.8)),
            ("Rule-based", P.rule_policy(0.02), COL["rule_based"]),
            ("Oracle", P.reward_max_oracle, COL["oracle"])]
    tr = {lab: run_episodes("full", f, [seed_env], trace=True) for lab, f, _ in pols}
    _style()
    fig, axs = plt.subplots(4, 1, figsize=(8, 9.2), sharex=True)
    t0 = tr[pols[0][0]]["traces"][0]
    t = np.arange(1, len(t0["vis"]) + 1)
    axs[0].plot(t, t0["vis"], color="#2E86C1", lw=2, label="visibility $V$ (km)")
    ax2 = axs[0].twinx(); ax2.plot(t, t0["logcn2"], color="#922B21", lw=1.6, ls="--", label=r"$\log_{10}C_n^2$")
    axs[0].set_ylabel("$V$ (km)"); ax2.set_ylabel(r"$\log_{10}C_n^2$")
    h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    for lab, _, st in pols:
        d = tr[lab]["traces"][0]
        axs[1].plot(t, d["tx"], label=f"{lab}  (return {tr[lab]['return'][0]:.1f})", **{**st, "lw": 1.6})
        axs[2].plot(t, 100 * d["w0"], **{**st, "lw": 1.6})
        axs[3].plot(t, d["snr"], **{**st, "lw": 1.6})
    axs[3].axhline(12, color="grey", lw=0.8); axs[3].axhline(6, color="red", lw=0.8)
    axs[1].set_ylabel("$P_t$ (dBm)"); axs[2].set_ylabel("$w_0$ (cm)"); axs[3].set_ylabel("SNR (dB)")
    axs[3].set_xlabel("Step"); axs[1].legend(fontsize=8)
    for i, ax in enumerate(axs):
        ax.set_title(f"({'abcd'[i]})", loc="left", fontsize=11)
    save(fig, "fig_rev_adapt.png")
    return {lab: dict(ret=float(tr[lab]["return"][0]), mean_tx=float(tr[lab]["mean_tx_dbm"][0]),
                      mean_w0_cm=float(100 * tr[lab]["mean_w0_m"][0]),
                      corr_tx_vis=float(np.corrcoef(tr[lab]["traces"][0]["tx"], t0["vis"])[0, 1])
                      if np.std(tr[lab]["traces"][0]["tx"]) > 0 else None) for lab, _, _ in pols}


def main():
    summ = json.loads((RES / "summary.json").read_text())["policies"]
    out = {}
    fig_main(summ)
    out["val_reference"] = fig_curves(None)
    fig_po(summ)
    out["adapt_episode"] = fig_adapt()
    (RES / "figures_info.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
