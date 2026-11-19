"""(F) Compare FSOEnv's empirical beam-wander sigma with the Andrews-Phillips
collimated-beam result <r_c^2> = 2.42 Cn2 L^3 W0^(-1/3) (Kolmogorov spectrum,
infinite outer scale, constant Cn2; Recolons, Andrews & Phillips, Opt. Eng. 46(8)
086002 (2007), Eq. (3) with 2F1 = 1; Andrews & Phillips, Laser Beam Propagation
through Random Media, 2nd ed., SPIE Press 2005).

FSOEnv draws x, y ~ N(0, sigma^2) per axis, so its <r_c^2> = 2 sigma^2.
Usage: python experiments/revision/beam_wander_calibration.py
Writes results/revision/beam_wander_calibration.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import REPO, FSOEnv, wander_gain
from physics.fso_channel import gaussian_beam_radius
from physics.turbulence_model import rytov_variance

L = FSOEnv.DISTANCE_M
LAM = FSOEnv.WAVELENGTH_M


def sim_sigma(log_cn2):
    strength = float(np.clip((log_cn2 + 15.4) / 1.5, 0.0, 1.0))
    return 0.004 + 0.055 * strength


def ap_rc_rms(cn2, w0):
    return float(np.sqrt(2.42 * cn2 * L ** 3 * w0 ** (-1.0 / 3.0)))


def main():
    rows = []
    for lc in (-15.5, -15.3, -15.1, -14.8, -14.4, -14.1):
        cn2 = 10 ** lc
        s = sim_sigma(lc)
        row = dict(log10_cn2=lc, rytov_var=float(rytov_variance(cn2, L, LAM)),
                   sim_rc_rms_cm=100 * np.sqrt(2) * s)
        for w0 in (0.01, 0.02, 0.035):
            ap = ap_rc_rms(cn2, w0)
            w = gaussian_beam_radius(w0, L, LAM)
            sim_loss = -10 * np.log10(wander_gain(cn2, w))
            ap_loss = -10 * np.log10(1.0 / (1.0 + 2.0 * ap ** 2 / w ** 2))   # 4 sigma^2/w^2, sigma^2 = <r^2>/2
            row[f"ap_rc_rms_cm_w0_{w0}"] = 100 * ap
            row[f"ratio_sim_over_ap_w0_{w0}"] = np.sqrt(2) * s / ap
            row[f"sim_wander_loss_db_w0_{w0}"] = sim_loss
            row[f"ap_wander_loss_db_w0_{w0}"] = ap_loss
        rows.append(row)
    out = REPO / "results" / "revision" / "beam_wander_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: round(float(v), 4) for k, v in r.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
