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

## Logged last-50-episode means

Computed from the committed `results/multi_algo_data.npz` as the mean of the
last 50 episodes, averaged over the stored seeds. No other summary was used.

| Policy | Last 50 episodes |
|---|---:|
| TD3 | 78.76 |
| SAC | 73.64 |
| PPO | 73.48 |
| Fixed `[10 dBm, 0.02 m]` | 68.31 |
| Random | 63.29 |

An earlier 400-episode array, in which the TD3 mean was about 87, was
overwritten by this 600-episode file. It is not in the repository.
