# Notes matching the committed comparison figure

The first part of this file describes the committed 3-seed / 600-episode run.
The section "Strong protocol" at the end describes the newer, stronger
protocol that `experiments/train_multi_seed.py` runs by default.

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
0.1 on every action coordinate. Correction: Stable-Baselines3 adds this noise
to the action after rescaling it to [-1, 1], so 0.1 is 5% of each
coordinate's half-range (0.6 dBm for power, 0.00125 m for waist), not the same
absolute value for power and waist as this note said before.

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
| Random | 63.19340231718413 |

Mean over every stored episode: TD3 82.52695307, SAC 75.62813292888889,
PPO 72.92858565166667, fixed 70.16166920216642, random 62.73962157034635.

TD3 last-50 means by seed are 71.4156208, 83.31231168, and 81.56262292.
The seed-averaged TD3 curve reaches 89.647805 at one episode index. That
single point is not the last-50 mean.

A previous array with shape `(3, 400)`, whose TD3 mean was about 87.1 and
whose last-20 mean was about 87.8, was overwritten when the 600-episode run
saved this file. No copy of that 400-episode array was found on disk.

## Strong protocol (default of `experiments/train_multi_seed.py`)

This protocol reports both training-episode returns and a held-out greedy
evaluation. The environment (3 km, 80 steps, reward above) and the
hyperparameters above are unchanged.

Command (single command, produces all data, summary tables, and figures):

```bash
python experiments/train_multi_seed.py
```

Defaults: TD3, SAC, PPO; training seeds 100–109 (`--seeds 10 --seed-base 100`);
1000 training episodes per run (`--episodes`, 600 also fine); 50 held-out
evaluation episodes per run (`--eval-episodes 50`).

### Training

Identical to the description above, with the extra explicit seeding listed
under "Seeding". Training returns are the raw Monitor episode sums.

### Held-out evaluation

- After training, the final model of each run is evaluated with
  `model.predict(obs, deterministic=True)`: the actor output for TD3, the
  tanh-squashed mean for SAC, and the clipped Gaussian mean for PPO.
- Observations get the same [-1, 1] map used during training.
- Training seed index `i` (seed `100 + i`) is evaluated on env seeds
  `eval_seed_base + 50*i + k`, k = 0..49, with `eval_seed_base = 1_000_000`.
  Training resets use only the training seed and the env's own continuing
  RNG, so these env seeds are never passed to a training reset.
- Each evaluation episode is one `env.reset(seed=s)` followed by 80 steps.
  Logged per episode: return, mean transmit power (dBm), mean SNR (dB), and
  outage fraction (steps with SNR < 6 dB, the env's `outage` flag).
- The env's RNG consumption does not depend on the action (one weather draw
  and one SNR-noise draw per step), so on a given env seed all policies see
  identical weather and noise. Comparisons per seed index are paired.

### Baselines (same evaluation function, same env seeds)

- Fixed: `[10 dBm, 0.02 m]` every step.
- Random: `env.action_space.sample()`, action space seeded with
  `eval_seed_base + 7919 * (i + 1)` for seed index `i`.
- Rule-based adaptive: with the raw observation
  `[visibility_km, log10(Cn2), rx_power_dbm, snr_db]`,
  if `snr_db < 10` or `visibility_km < 3` the action is `[15 dBm, 0.02 m]`,
  otherwise `[10 dBm, 0.02 m]`. `snr_db` in the observation is the SNR
  measured on the previous step (after the reset it is the noise-free SNR of
  the unrewarded `[10 dBm, 0.02 m]` probe action). Visibility is the current
  value. Actions are in physical units, which is the env's action space, and
  are clipped to the box.
- Code: `experiments/evaluation.py`.

### Statistics reported

- Held-out: for each policy and seed index, the mean over 50 episodes; then
  the mean, sample std (ddof=1), and SE = std/sqrt(n) across seed indices.
  The std over all 500 episodes is also in the CSV.
- Training: the mean of the last 50 training episodes per seed, then the mean
  and std across seeds, and the mean over all training episodes.
- Learning-curve figure: each seed's training curve gets a Gaussian smooth
  (width 50), then the mean ± std (or ± SE with `--band se`) across seeds.
  Baselines are drawn as horizontal lines at their held-out evaluation mean.

### Seeding

`seed_everything` seeds Python `random`, NumPy's global RNG, and PyTorch.
SB3's `set_random_seed` is also called. The training env is reset with the
run seed, and its action and observation spaces are seeded. SB3 also reseeds
the env and action space from `seed`. Each run uses one PyTorch thread. On
the same machine and library versions, two runs of the same command gave
identical arrays (the per-episode mean SNR differed by at most ~4e-15 from float rounding; all returns were bit-identical).

### Output

`results/multi_algo_strong.npz` holds all raw per-seed arrays; see README.md
for the keys. `results/multi_algo_strong_summary.{csv,md}`,
`results/strong_learning_curves.png`, `results/strong_eval_bar.png`,
`results/strong_eval_box.png`.

### Numbers

Not available yet. The full strong protocol has not been run. Only a tiny
smoke test (2 seeds, 12 episodes, 3 eval episodes) was run to check that the
pipeline runs end to end. Its numbers are not results and are not reported.

### Old result

`python experiments/train_multi_seed.py --preset legacy` re-runs the 3-seed /
600-episode protocol and writes `results/multi_algo_data_rerun.npz`.
`--preset legacy-committed --plot-only` redraws the committed figure from
the committed `results/multi_algo_data.npz`.

### Notes on the environment (not changed)

- `step` returns `terminated=True` at the 80-step limit instead of
  `truncated=True`. A time limit is normally a truncation, and SB3 would then
  bootstrap the final value. The observation has no time feature, so the
  original behaviour is kept to stay comparable with the committed results.
  Treat this as a known modelling choice.
- `experiments/train_sac.py` and `experiments/compare_baseline.py` (the
  earlier SAC-only checkpoint) use `max_steps=100`, not 80. They are not part
  of either protocol.

