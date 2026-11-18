## Strong Protocol Results (Recommended)

The default experiment uses a stronger and more rigorous protocol:

- **10 independent seeds** (100–109)
- **1000 training episodes** per algorithm
- **50 held-out greedy evaluation episodes** after training (deterministic policy)
- Comparison against three baselines: Fixed, Random, and a simple **Rule-based** adaptive policy

### Held-out Evaluation Results
(mean ± std across 10 seeds)

| Policy              | Eval Return     | Outage Fraction | Mean Tx Power (dBm) |
|---------------------|-----------------|-----------------|---------------------|
| **Rule-based**      | **82.99 ± 0.44**| **0.000**       | 12.72               |
| SAC                 | 76.26 ± 4.52    | 0.070           | 11.52               |
| PPO                 | 72.33 ± 11.70   | 0.041           | 10.27               |
| TD3                 | 72.13 ± 8.23    | 0.132           | 11.36               |
| Fixed [10 dBm, 0.02 m] | 69.28 ± 1.23 | 0.012           | 10.00               |
| Random              | 62.07 ± 0.92    | 0.232           | 10.01               |

**Key observation**: A simple rule-based policy (increase power to 15 dBm when SNR < 10 dB or visibility < 3 km) outperforms all three deep RL algorithms in this environment.

### How to Reproduce
```bash
# Strong protocol (10 seeds, 1000 episodes, held-out eval + rule-based baseline)
python experiments/train_multi_seed.py

# Or explicitly
python experiments/train_multi_seed.py --preset strong
