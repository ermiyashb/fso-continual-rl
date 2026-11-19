# Weather-adaptive free-space optical link control

This repository is the code for the weather-adaptive FSO deep-RL comparison.
The three-algorithm figure comes from `environment/fso_env.py` and
`experiments/train_multi_seed.py`, with the logged returns in
`results/multi_algo_data.npz`.

Two protocols are supported by `experiments/train_multi_seed.py`:

- **strong** (default): 10 seeds, 1000 training episodes, and a separate
  50-episode held-out greedy evaluation of each final policy against Fixed,
  Random, and Rule-based baselines. Both training returns and held-out
  evaluation returns are reported.
- **legacy** (`--preset legacy`): the earlier 3-seed / 600-episode protocol
  that reports training-episode returns only. The committed numbers further
  down come from that protocol.

`experiments/train_sac.py` and `results/sac_fso.zip` are an earlier SAC-only
checkpoint. They are not the source of the TD3 / SAC / PPO numbers below.

## Environment

The link is 3 km at 1550 nm. Each episode has 80 steps.

| | |
|---|---|
| Observation | visibility (km), log10(Cn2), previous received power (dBm), previous SNR (dB) |
| Action | transmit power in [4, 16] dBm, launch beam waist in [0.01, 0.035] m |
| Reward | `r = (100 / 80) * (q - 0.16 * (P_tx - 4) / 12)` with `q = clip((SNR_dB - 1) / 11, 0, 1)` |

Fog follows the Kim model. Geometric capture is that of a Gaussian beam on a
2.5 cm receiver. The Gamma–Gamma draw inside the channel routine is removed
and replaced by a deterministic beam-wander factor that increases with Cn2.
SNR noise has standard deviation 0.15 dB. The noise floor is −12 dBm.
Details and the hyperparameters are in `METHODS_NOTES.md`.

## Strong protocol (held-out evaluation, 10 seeds, rule-based baseline)

`experiments/train_multi_seed.py` now defaults to a stronger protocol. It
reports **both** the training-episode returns **and** a separate held-out
greedy evaluation of each final policy:

- TD3, SAC, and PPO, each trained on 10 seeds (100–109) for 1000 episodes
  (configurable with `--episodes`, `--seeds`, `--seed-base`).
- After training, the final policy of every run is evaluated greedily
  (`deterministic=True`) for 50 episodes (`--eval-episodes`) on held-out
  environment seeds (from 1,000,000 up, `--eval-seed-base`) that are never
  used for training. Training seed index `i` uses env seeds
  `1_000_000 + 50*i ... 1_000_000 + 50*i + 49`.
- Three non-learning baselines are scored with the **same** evaluation
  function on the **same** held-out env seeds:
  - **Fixed**: always `[10 dBm, 0.02 m]`.
  - **Random**: uniform sample from the action box (action space seeded).
  - **Rule-based adaptive**: if the current SNR < 10 dB or the visibility
    < 3 km, use `[15 dBm, 0.02 m]`; otherwise use `[10 dBm, 0.02 m]`.
    "Current SNR" is the SNR entry of the observation, i.e. the SNR measured
    on the previous step (the most recent value a transmitter can know); the
    visibility entry is the current visibility. Both actions are already in
    the env's physical action units.
- The environment's random-number use does not depend on the action, so every
  policy sees exactly the same weather track and SNR noise on a given env
  seed. Per-seed comparisons are therefore paired.

### Reproduce (one command)

From the repository root, with the packages in `requirements.txt`:

```bash
python experiments/train_multi_seed.py
```

This trains 3 algorithms x 10 seeds x 1000 episodes (runs in parallel, one
CPU thread per run), evaluates everything, and writes:

| File | Content |
|---|---|
| `results/multi_algo_strong.npz` | all raw per-seed data (see below) |
| `results/multi_algo_strong_summary.csv` / `.md` | summary table computed from the npz |
| `results/strong_learning_curves.png` | training returns, mean ± std across seeds |
| `results/strong_eval_bar.png` | held-out evaluation, mean ± std across seeds, per-seed dots |
| `results/strong_eval_box.png` | held-out evaluation, box plot of per-seed means |

Useful variants:

```bash
python experiments/train_multi_seed.py --episodes 600          # same protocol, 600 training episodes
python experiments/train_multi_seed.py --band se               # plot standard error instead of std
python experiments/train_multi_seed.py --save-models           # also keep the 30 final models
python experiments/train_multi_seed.py --plot-only             # redraw figures + summary from the npz
python experiments/train_multi_seed.py --plot-only --band se   # same, with SE bands
```

Do not pass `--fast` for numbers that should match the logged update schedule.

Contents of `results/multi_algo_strong.npz`:

- `train_TD3`, `train_SAC`, `train_PPO`: `(n_seeds, episodes)` training-episode returns.
- `eval_<policy>`: `(n_seeds, eval_episodes)` held-out greedy returns for
  `TD3`, `SAC`, `PPO`, `fixed`, `random`, `rule_based`; plus
  `eval_<policy>_tx` (mean transmit power, dBm), `eval_<policy>_snr`
  (mean SNR, dB), and `eval_<policy>_outage` (fraction of steps with SNR < 6 dB).
- `train_seeds`, `eval_env_seeds`, `episodes`, `eval_episodes`,
  `eval_seed_base`, `max_steps`, `fast`, `config_json`.

In the summary, "± std" is the standard deviation across seeds of each
seed's mean (sample std, ddof=1); SE = std / sqrt(n_seeds). For the fixed and
rule-based baselines the spread across seeds comes only from the different
held-out env seeds of each seed index.

### Results of the strong protocol

*Not run yet.* Fill this table from `results/multi_algo_strong_summary.md`
after running the command above. Do not fill it from smoke tests.

| Policy | Held-out eval return (mean ± std, 10 seeds x 50 ep.) | Training return, last 50 ep. (mean ± std) |
|---|---:|---:|
| TD3 | TBD | TBD |
| SAC | TBD | TBD |
| PPO | TBD | TBD |
| Fixed `[10 dBm, 0.02 m]` | TBD | n/a |
| Random | TBD | n/a |
| Rule-based | TBD | n/a |

## Old 3-seed / 600-episode comparison (committed result)

The sections below describe the earlier, committed run. It is kept as is.

### Reproduce the old comparison
From the repository root, with the packages in `requirements.txt`:

```powershell
python experiments/train_multi_seed.py --preset legacy
```

That command trains SAC, PPO, and TD3 on seeds 100, 101, and 102 for 600
episodes, rolls out the fixed action `[10 dBm, 0.02 m]` (env seeds 999...)
and a uniform random policy (env seeds 888...), and writes
`results/multi_algo_data_rerun.npz` and `results/multi_algo_comparison_rerun.png`.
It no longer overwrites the committed `results/multi_algo_data.npz`.
Do not pass `--fast` if you want the same update schedule as the logged run.
The random rollout now seeds the action space (seed 888); the committed
random curve was drawn with an unseeded action space, so that one curve
cannot be reproduced bit for bit. Learner curves can still differ slightly
across library versions and hardware.

To redraw the committed figure from the committed array:

```powershell
python experiments/train_multi_seed.py --preset legacy-committed --plot-only
```

The curves are training-episode returns. For SAC, PPO, and TD3 the plot shows
a Gaussian smooth (width 50) of the mean across the three seeds. They are not
a separate greedy evaluation after training. The fixed and random curves are
separate 600-episode rollouts (seeds 999 and 888).

### Logged training-episode returns

Computed from the current `results/multi_algo_data.npz`, the latest
600-episode run. SAC, PPO, and TD3 were each trained on seeds 100, 101, and
102; the learner figures are the mean across those three seeds. Fixed and
random are the single stored 600-episode rollouts. These are the returns
collected during training, not a later greedy evaluation.

| Policy | Last 50 episodes | All episodes |
|---|---:|---:|
| TD3 | 78.76 | 82.53 |
| SAC | 73.64 | 75.63 |
| PPO | 73.48 | 72.93 |
| Fixed `[10 dBm, 0.02 m]` | 68.31 | 70.16 |
| Random | 63.19 | 62.74 |

TD3 last-50 means on the three seeds are 71.42, 83.31, and 81.56. The
array shape is `(3, 600)` for each learner and `(600,)` for fixed and random.
