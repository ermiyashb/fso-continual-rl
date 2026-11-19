"""Write LaTeX table bodies for the manuscript from the revision outputs.
Usage: python experiments/revision/make_tables.py [out_dir]
Only reads results/revision/*.json|csv|npz; tables whose inputs are missing get a 'pending' row.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import REPO

RES = REPO / "results" / "revision"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "tables"


def pf(p):
    if p < 1e-6:
        return "$<10^{-6}$"
    if p >= 0.995:
        return "1.00"
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def sg(x, d=2):
    return f"${x:+.{d}f}$"


def ci(c, d=2):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def read_csv(name):
    f = RES / f"stats_{name}.csv"
    if not f.exists():
        return None
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = json.loads(v)
            except Exception:
                pass
    return rows


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def paired_table(name, rows, ncols=7):
    if rows is None:
        write(name, [r"\multicolumn{%d}{c}{(pending)}\\" % ncols]); return
    L = []
    for r in rows:
        L.append(f"{r['comparison']} & {sg(r['diff'])} & {ci(r['diff_ci'])} & {r['t']:.2f} & {pf(r['p_t_holm'])} "
                 f"& {pf(r['p_w_holm'])} & {r['higher']}/10\\\\")
    write(name, L)


LBL = {
    "orig_TD3": "TD3", "orig_SAC": "SAC", "orig_PPO": "PPO", "orig_fixed": "Fixed [10 dBm, 2 cm]",
    "orig_random": "Random", "orig_rule_based": "Rule-based",
    "fixed_10_0.02": "Fixed [10 dBm, 2 cm]", "fixed_grid_best": "Best fixed$^{a}$",
    "rule_w0.02": "Rule-based", "rule_w0.035": "Rule-based, $w_0=3.5$ cm",
    "oracle_minpower": "Min.-power oracle$^{b}$", "oracle_rewardmax": "Reward-max.\\ oracle$^{b}$",
}


def policy_row(s, label, w0=True, extra=""):
    w = f"{s['mean_w0_cm']:.2f}" if (w0 and s.get("mean_w0_cm") is not None) else "--"
    return (f"{label} & ${s['mean']:.2f}\\pm{s['sd']:.2f}$ & {s['iqm']:.2f} {ci(s['iqm_ci'])} & {s['outage']:.3f} "
            f"& {s['mean_tx_dbm']:.2f} & {w}{extra}\\\\")


def main():
    sj = RES / "summary.json"
    S = json.loads(sj.read_text()) if sj.exists() else {"policies": {}, "selection": {}}
    P = S["policies"]

    # original protocol: IQM table
    L = [policy_row(P[k], LBL[k], w0=False) for k in
         ("orig_TD3", "orig_SAC", "orig_PPO", "orig_fixed", "orig_random", "orig_rule_based") if k in P]
    write("tab_orig_iqm", L or [r"\multicolumn{6}{c}{(pending)}\\"])
    paired_table("tab_orig_paired", read_csv("original"))
    w = read_csv("original_welch")
    write("tab_orig_welch", [f"{r['comparison']} & {sg(r['diff'])} & {r['t']:.2f} & {r['df']:.1f} & {pf(r['p'])} & {pf(r['p_holm'])} & {r['higher']}/10\\\\"
                             for r in w] if w else [r"\multicolumn{7}{c}{(pending)}\\"])

    # non-learned
    L = [policy_row(P[k], LBL[k]) for k in ("fixed_10_0.02", "fixed_grid_best", "rule_w0.02", "rule_w0.035",
                                            "oracle_minpower", "oracle_rewardmax") if k in P]
    write("tab_nonlearned", L or [r"\multicolumn{6}{c}{(pending)}\\"])

    # corrected protocol (full observation)
    L = []
    for a in ("TD3", "SAC", "PPO"):
        for tag, tl in (("final", "final"), ("best", "best-val.")):
            k = f"full_{a}_{tag}"
            if k in P:
                sel = S["selection"].get(f"full_{a}", {})
                be = sel.get("best_val_episode")
                ex = f" & {int(np.median(be))}" if (be and tag == "best") else " & --"
                L.append(policy_row(P[k], f"{a} ({tl})", extra=ex))
    write("tab_corrected", L or [r"\multicolumn{7}{c}{(pending)}\\"])
    paired_table("tab_corrected_best", read_csv("corrected_best"))
    paired_table("tab_corrected_final", read_csv("corrected_final"))
    paired_table("tab_best_vs_final", read_csv("best_vs_final"))
    paired_table("tab_corr_vs_orig", read_csv("corrected_vs_original"))

    # partial observability
    L = []
    for v in ("po_k0", "po_k1", "po_k3"):
        k = v[-1]
        rows = []
        for a, al in (("SAC", "SAC"), ("TD3", "TD3"), ("PPO", "PPO"), ("RPPO", "Recurrent PPO")):
            for tag, tl in (("best", "best-val."), ("final", "final")):
                if f"{v}_{a}_{tag}" in P:
                    rows.append(policy_row(P[f"{v}_{a}_{tag}"], f"$k={k}$: {al} ({tl})"))
        for b, bl in (("snr_rule_w0.02", "SNR rule, 2 cm"), ("snr_rule_w0.035", "SNR rule, 3.5 cm"),
                      ("gain_tracking", "Gain tracking$^{c}$")):
            if f"{v}_{b}" in P:
                rows.append(policy_row(P[f"{v}_{b}"], f"$k={k}$: {bl}"))
        if rows:
            if L:
                L.append(r"\midrule")
            L += rows
    write("tab_po", L or [r"\multicolumn{6}{c}{(pending)}\\"])
    L = []
    for k in ("k0", "k1", "k3"):
        rows = read_csv(f"po_best_{k}")
        if rows:
            if L:
                L.append(r"\midrule")
            L += [f"{r['comparison']} & {sg(r['diff'])} & {ci(r['diff_ci'])} & {r['t']:.2f} & {pf(r['p_t_holm'])} "
                  f"& {pf(r['p_w_holm'])} & {r['higher']}/10\\\\" for r in rows]
    write("tab_po_tests", L or [r"\multicolumn{7}{c}{(pending)}\\"])

    # adaptation
    af = RES / "adaptation_metrics.json"
    if af.exists():
        A = json.loads(af.read_text())
        nm = {"oracle_rewardmax": "Reward-max.\\ oracle", "rule_based": "Rule-based", "rule_w0.035": "Rule-based, 3.5 cm",
              "fixed_grid_best": "Best fixed", "full_TD3_bestval": "TD3 (best-val.)", "full_SAC_bestval": "SAC (best-val.)",
              "full_PPO_bestval": "PPO (best-val.)", "full_TD3_final": "TD3 (final)", "full_SAC_final": "SAC (final)",
              "full_PPO_final": "PPO (final)"}
        L = [f"{nm[k]} & {A[k]['corr_P_Pstar']:.2f} & {A[k]['rms_P_minus_Pstar_db']:.2f} & {A[k]['corr_P_visibility']:.2f} "
             f"& {A[k]['mean_P']:.2f} & {A[k]['mean_w0_cm']:.2f} & {A[k]['frac_w0_ge_3cm']:.2f}\\\\" for k in nm if k in A]
        write("tab_adapt", L)
    else:
        write("tab_adapt", [r"\multicolumn{7}{c}{(pending)}\\"])

    # beam wander
    bw = RES / "beam_wander_calibration.csv"
    with open(bw) as fh:
        rows = list(csv.DictReader(fh))
    L = []
    for r in rows:
        f = {k: float(v) for k, v in r.items()}
        L.append(f"${f['log10_cn2']:.1f}$ & {f['rytov_var']:.2f} & {f['sim_rc_rms_cm']:.2f} & {f['ap_rc_rms_cm_w0_0.02']:.2f} & "
                 f"{f['ap_rc_rms_cm_w0_0.035']:.2f} & {f['ratio_sim_over_ap_w0_0.02']:.2f}--{f['ratio_sim_over_ap_w0_0.035']:.2f} & "
                 f"{f['sim_wander_loss_db_w0_0.035']:.2f} / {f['ap_wander_loss_db_w0_0.035']:.2f}\\\\")
    write("tab_beamwander", L)
    # number macros: \Rv{policy key}{field}, \Sv{family}{row index}{field}
    M = []
    def d(name, val):
        M.append("\\expandafter\\def\\csname %s\\endcsname{%s}" % (name, val))
    for k, s in P.items():
        d(f"r@{k}@mean", f"{s['mean']:.2f}"); d(f"r@{k}@sd", f"{s['sd']:.2f}")
        d(f"r@{k}@iqm", f"{s['iqm']:.2f}"); d(f"r@{k}@iqmci", ci(s['iqm_ci']))
        d(f"r@{k}@out", f"{s['outage']:.3f}"); d(f"r@{k}@tx", f"{s['mean_tx_dbm']:.2f}")
        d(f"r@{k}@outmax", f"{s['outage_seed_max']:.3f}"); d(f"r@{k}@pfive", f"{s['p05_episode_return']:.1f}")
        d(f"r@{k}@min", f"{min(s['per_seed']):.2f}"); d(f"r@{k}@max", f"{max(s['per_seed']):.2f}")
        if s.get("mean_w0_cm") is not None:
            d(f"r@{k}@wz", f"{s['mean_w0_cm']:.2f}")
    for fam in ("original", "corrected_best", "corrected_final", "best_vs_final", "corrected_vs_original", "po_best_k0", "po_best_k1", "po_best_k3"):
        rows = read_csv(fam)
        for i, r in enumerate(rows or []):
            d(f"s@{fam}@{i}@diff", f"{r['diff']:+.2f}"); d(f"s@{fam}@{i}@ci", ci(r['diff_ci']))
            d(f"s@{fam}@{i}@pt", pf(r['p_t_holm'])); d(f"s@{fam}@{i}@pw", pf(r['p_w_holm']))
            d(f"s@{fam}@{i}@ptraw", pf(r['p_t'])); d(f"s@{fam}@{i}@hi", f"{r['higher']}")
            d(f"s@{fam}@{i}@lab", r['comparison'])
    for k, v in S.get("selection", {}).items():
        be = v["best_val_episode"]
        d(f"sel@{k}@med", f"{int(np.median(be))}"); d(f"sel@{k}@min", f"{min(be)}"); d(f"sel@{k}@max", f"{max(be)}")
        d(f"sel@{k}@first", f"{v['train_first50']:.2f}"); d(f"sel@{k}@last", f"{v['train_last50']:.2f}")
    af = RES / "adaptation_metrics.json"
    if af.exists():
        for k, v in json.loads(af.read_text()).items():
            for f, val in v.items():
                d(f"a@{k}@{f.replace('_', '')}", f"{val:.2f}")
    (OUT / "numbers.tex").write_text("\n".join(M) + "\n")
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
