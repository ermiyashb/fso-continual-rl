# Replacement text for the manuscript

Use these passages in place of the current Abstract, Section 3, Table 1, Section 4.1, and Conclusion. They report only the committed `results/multi_algo_data.npz` from the 600-episode run. The earlier TD3 value near 87.3 came from a 400-episode file that was overwritten and is not in the repository. Do not cite it.

The logged curves are training-episode returns, averaged over 3 seeds (100, 101, and 102). They are not a separate greedy evaluation.

## Abstract

We study continuous control of a 3 km free-space optical link, in which the agent sets transmit power and launch beam waist while visibility and the refractive-index structure constant drift within each 80-step episode. SAC, PPO, and TD3 are trained for 600 episodes on three seeds with Stable-Baselines3. Reported values are training-episode returns, averaged over those seeds, not a held-out greedy test. Over the last 50 episodes the means are 78.76 for TD3, 73.64 for SAC, and 73.48 for PPO, compared with 68.31 for a fixed action of 10 dBm and 0.02 m and 63.19 for a uniform random policy. TD3 is the highest of the three learners. The margin over the fixed policy is modest, about ten return points, and SAC and PPO finish only a few points above that baseline.

## 3. Methods

### 3.1 Link and episode

The simulator is a 3 km, 1550 nm link. Received power uses Kim fog attenuation, the geometric capture of a Gaussian beam by a receiver of radius 2.5 cm, and transmit and receive optical efficiencies of 0.8. A Gamma-Gamma sample drawn inside the channel routine is divided out and replaced by its mean. Turbulence enters as a deterministic beam-wander factor, 1 / (1 + 4 sigma^2 / w^2), where sigma increases with log10(Cn2). Gaussian noise of standard deviation 0.15 dB is added to the SNR. The noise floor is -12 dBm.

Each episode has 80 steps. The action is transmit power in [4, 16] dBm and launch beam waist in [0.01, 0.035] m. The observation is visibility in kilometres, log10(Cn2), and the received power and SNR from the latest action. Weather follows five interpolated waypoints. Visibility stays about 1.15-9 km and log10(Cn2) about -15.6 to -14.0.

### 3.2 Reward

With transmit power P_tx in dBm,

q = clip((SNR_dB - 1) / 11, 0, 1)

r = 1.25 * (q - 0.16 * (P_tx - 4) / 12)

The factor 1.25 equals 100/80. There is no term that rewards a beam waist of 0.02 m. The logged score is the sum of the 80 step rewards.

### 3.3 Training protocol

SAC, PPO, and TD3 are trained with Stable-Baselines3 2.9.2 and PyTorch 2.15.0 on CPU. Observations are scaled into [-1, 1] using the observation bounds. The stored returns are the unscaled episode sums.

SAC and TD3 use an MlpPolicy, learning rate 3e-4, batch size 256, and a learning start of 500 steps. Remaining hyperparameters are the library defaults. The reduced-update flag was not used. TD3 adds Gaussian action noise with standard deviation 0.1 on both action coordinates. PPO uses learning rate 2.5e-4, 480 steps per update, batch size 96, 10 epochs, discount 0.99, GAE lambda 0.95, and clip range 0.15, with policy and value networks of two layers of 128 units. Its action mean is initialized at the center of the action box, and each coordinate's exploration standard deviation is 8 percent of that coordinate's range.

Each learner is trained for 600 episodes on seeds 100, 101, and 102. The fixed baseline repeats [10 dBm, 0.02 m] for 600 episodes with seed 999. The random baseline samples the action box uniformly for 600 episodes with seed 888. Figures show a Gaussian smooth of width 50 applied to the mean across the three training seeds.

## Table 1

Mean training-episode return over the last 50 episodes. Learning algorithms are averaged over 3 seeds.

| Policy | Last-50 mean |
|---|---:|
| TD3 | 78.76 |
| SAC | 73.64 |
| PPO | 73.48 |
| Fixed (10 dBm, 0.02 m) | 68.31 |
| Random | 63.19 |

## 4.1 Learning comparison

Table 1 is computed from the 600-episode log. All three learners finish above the fixed and random policies. The ordering on the last 50 episodes is TD3, then SAC, then PPO. The gaps are small relative to the scale of the return: TD3 is 10.45 above the fixed policy and 5.12 above SAC, while SAC and PPO differ by 0.16. These values are training returns averaged over 3 seeds, not a separate greedy evaluation. A single earlier 400-episode run that placed TD3 near 87.3 was overwritten and is not part of this table.

## Conclusion

On the surviving 600-episode log, TD3 obtains the highest last-50 training return (78.76), followed by SAC (73.64) and PPO (73.48). All three exceed the fixed policy (68.31) and the random policy (63.19). The improvement over the fixed action is real but modest, and it is measured on training episodes averaged over three seeds. The result does not by itself establish a large or uniform advantage for any one algorithm.
