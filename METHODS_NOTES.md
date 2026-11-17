# Notes matching the committed comparison figure

These notes describe the code and the file that produced
`results/multi_algo_comparison.png`. They are not a second experiment.
The array is `results/multi_algo_data.npz`.

Shapes: `SAC`, `PPO`, and `TD3` are `(3, 600)`. `fixed` and `random` are
`(600,)`. Learning seeds are 100, 101, and 102. The fixed rollout uses seed
999 and the random rollout uses seed 888. Training did not use `--fast`.

## Channel and episode

- Distance 3000 m, wavelength 1550 nm, receiver radius 0.025 m.
- Optical efficiencies `eta_tx = eta_rx = 0.8` (defaults of `simulate_fso_channel`).
- Kim fog transmission. Gaussian-beam geometric capture.
- The Gamma–Gamma sample is divided out. Turbulence is the expected wander
  gain `1 / (1 + 4 sigma^2 / w_rx^2)`, with
  `sigma = 0.004 + 0.055 * clip((log10(Cn2) + 15.4) / 1.5, 0, 1)`.
- SNR noise is Gaussian with standard deviation 0.15 dB.
- Noise floor −12 dBm. SNR target used only inside the reward is not a
  separate threshold: quality uses `(SNR_dB - 1) / 11`.
- Episode length 80. The environment default `max_steps` is 80, and the
  training script also passes `max_steps=80`.

Weather is five interpolated waypoints drawn from this catalog, then clipped:

- visibility about 1.15–9 km
- `log10(Cn2)` about −15.6 to −14.0

The catalog rows are `(visibility_km, log10(Cn2))`:

`(7.2, -15.3), (5.5, -14.3), (3.0, -15.1), (1.8, -14.4), (1.35, -15.2)`.

## Reward

Per step, with `P_tx` in dBm:

```text
q = clip((SNR_dB - 1) / 11, 0, 1)
r = (100 / 80) * (q - 0.16 * (P_tx - 4) / 12)
```

`100/80 = 1.25`. There is no beam-waist penalty toward 0.02 m.
The logged return is the sum of these 80 step rewards.

## Learning setup

Library: Stable-Baselines3 2.9.2, Gymnasium 1.4.0, PyTorch 2.15.0, CPU.
Observations are affinely mapped from the observation box into `[-1, 1]`.
The stored returns are the raw episode sums from the monitor, underneath
that wrapper.

Each algorithm runs for 600 episodes. The callback stops training at that
count. The `learn` call uses a step budget of `600 * 80 * 2`, which is only
an upper bound.

SAC and TD3: `MlpPolicy`, learning rate `3e-4`, batch size 256,
`learning_starts=500`. Other hyperparameters are the library defaults.
`--fast` was not set, so `train_freq` and `gradient_steps` stay at those
defaults. TD3 action noise is `NormalActionNoise` with standard deviation
0.1 on every action coordinate (the same absolute value for power and waist).

PPO: learning rate `2.5e-4`, `n_steps=480`, `batch_size=96`, `n_epochs=10`,
`gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.15`, `ent_coef=0`,
`vf_coef=0.5`, `max_grad_norm=0.5`, networks `pi=[128, 128]` and
`vf=[128, 128]`. After construction, the action-mean layer is zeroed and its
bias is set to the center of the action box. `log_std` is set to
`log(0.08 * (high - low))` per coordinate.

## What the figure is

`experiments/train_multi_seed.py` plots a Gaussian smooth of width 50 of the
mean across the three training seeds. Those curves are the returns collected
while training. They are not a frozen greedy policy evaluated afterwards.
Fixed and random are independent rollouts, not horizontal lines fitted to
the learners.

## Numbers in the committed array

Last 50 episodes, mean over the stored runs:

| Policy | Value |
|---|---:|
| TD3 | 78.76351846666667 |
| SAC | 73.63630902 |
| PPO | 73.48153086 |
| Fixed | 68.3145986346532 |
| Random | 63.29205688020732 |

Mean over every stored episode: TD3 82.52695307, SAC 75.62813292888889,
PPO 72.92858565166667, fixed 70.16166920216642, random 62.72875264550862.

TD3 last-50 means by seed are 71.4156208, 83.31231168, and 81.56262292.
The seed-averaged TD3 curve reaches 89.647805 at one episode index. That
single point is not the last-50 mean.

A previous array with shape `(3, 400)`, whose TD3 mean was about 87.1 and
whose last-20 mean was about 87.8, was overwritten when the 600-episode run
saved this file. No copy of that 400-episode array was found on disk.
