| Policy | Held-out eval return (mean ± std across seeds) | SE | Outage frac | Mean P_tx (dBm) | Training return, last 50 episodes (mean ± std) |
|---|---:|---:|---:|---:|---:|
| TD3 | 72.13 ± 8.23 | 2.60 | 0.132 | 11.36 | 72.01 ± 8.31 |
| SAC | 76.26 ± 4.52 | 1.43 | 0.070 | 11.52 | 76.31 ± 4.08 |
| PPO | 72.33 ± 11.70 | 3.70 | 0.041 | 10.27 | 70.80 ± 11.87 |
| Fixed [10 dBm, 0.02 m] | 69.28 ± 1.23 | 0.39 | 0.012 | 10.00 | n/a (not trained) |
| Random | 62.07 ± 0.92 | 0.29 | 0.232 | 10.01 | n/a (not trained) |
| Rule-based | 82.99 ± 0.44 | 0.14 | 0.000 | 12.72 | n/a (not trained) |
