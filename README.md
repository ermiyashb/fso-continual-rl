# Weather-adaptive free-space optical link control

This repository is the code for the weather-adaptive FSO deep-RL comparison.
The three-algorithm figure comes from `environment/fso_env.py` and
`experiments/train_multi_seed.py`, with the logged returns in
`results/multi_algo_data.npz`.

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

## Reproduce the comparison

From the repository root, with the packages in `requirements.txt`:

```powershell
python experiments/train_multi_seed.py --episodes 600 --seeds 3
```

That command trains SAC, PPO, and TD3 on seeds 100, 101, and 102, evaluates
the fixed action `[10 dBm, 0.02 m]` and a uniform random policy, and writes
`results/multi_algo_data.npz` and `results/multi_algo_comparison.png`.
Do not pass `--fast` if you want the same update schedule as the logged run.

To redraw the figure from the committed array:

```powershell
python experiments/train_multi_seed.py --plot-only
```

The curves are training-episode returns. For SAC, PPO, and TD3 the plot shows
a Gaussian smooth (width 50) of the mean across the three seeds. They are not
a separate greedy evaluation after training. The fixed and random curves are
separate 600-episode rollouts (seeds 999 and 888).

## Logged training-episode returns

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
