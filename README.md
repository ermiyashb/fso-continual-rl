## Results

### Strong Protocol (10 seeds, held-out evaluation)

Default experiment settings:
- 10 seeds (100–109)
- 1000 training episodes
- 50 held-out greedy evaluation episodes after training
- Baselines: Fixed, Random, and Rule-based adaptive

**Held-out evaluation return (mean ± std across 10 seeds):**

| Policy | Eval Return | Outage Fraction | Mean Tx Power (dBm) |
|--------|-------------|-----------------|---------------------|
| Rule-based | 82.99 ± 0.44 | 0.000 | 12.72 |
| SAC | 76.26 ± 4.52 | 0.070 | 11.52 |
| PPO | 72.33 ± 11.70 | 0.041 | 10.27 |
| TD3 | 72.13 ± 8.23 | 0.132 | 11.36 |
| Fixed [10 dBm, 0.02 m] | 69.28 ± 1.23 | 0.012 | 10.00 |
| Random | 62.07 ± 0.92 | 0.232 | 10.01 |

A simple rule-based policy outperforms TD3, SAC, and PPO under the current environment and reward.

### Reproduce the strong results
```bash
python experiments/train_multi_seed.py
