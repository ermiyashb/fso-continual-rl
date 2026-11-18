# Changes: stronger, reproducible experimental protocol

The environment (`environment/fso_env.py`: 3 km, 80 steps, reward), the
physics, and the learner hyperparameters were **not** changed. No results
were added. The new protocol has not been run in full yet.

## New / changed files

- `experiments/train_multi_seed.py` (rewritten, backward compatible)
  - `--preset strong` (default): TD3, SAC, PPO; 10 seeds (100–109); 1000
    training episodes (`--episodes`, e.g. 600); then a separate **greedy
    (deterministic) evaluation of each final policy for 50 episodes**
    (`--eval-episodes`) on **held-out env seeds** (from `--eval-seed-base`
    1,000,000; disjoint per seed index).
  - Fixed, Random, and the new Rule-based baseline are evaluated with the
    same function on the same held-out env seeds (paired comparison).
  - Saves all raw per-seed data to `results/multi_algo_strong.npz`: training
    returns `(seeds, episodes)`; eval returns, mean P_tx, mean SNR, and outage
    fraction `(seeds, eval_episodes)` per policy; seeds; config.
  - Writes `results/multi_algo_strong_summary.csv` and `.md` (mean, std, and
    SE across seeds for held-out eval; last-50 and all-episode training
    returns).
  - Figures: `results/strong_learning_curves.png` (training, mean ± std or
    ± SE across seeds, baselines as held-out reference lines),
    `results/strong_eval_bar.png` (mean ± std with per-seed dots), and
    `results/strong_eval_box.png`.
  - `--preset legacy` reproduces the old 3-seed / 600-episode protocol and
    writes `*_rerun` files, so the committed `results/multi_algo_data.npz`
    is never overwritten. `--preset legacy-committed --plot-only` redraws
    the committed figure.
  - Other flags: `--plot-only`, `--band {std,se}`, `--save-models`,
    `--algos`, `--seed-base`, `--out`, `--workers`, `--fast`, `--show`.
    Figures are saved headless by default (no blocking `plt.show()`).
- `experiments/evaluation.py` (new): the held-out evaluation protocol, the
  Fixed / Random / Rule-based policies, a wrapper for SB3 models as greedy
  policies, and `seed_everything`.
  - Rule-based: if the observed SNR < 10 dB or the visibility < 3 km, use
    `[15 dBm, 0.02 m]`, otherwise `[10 dBm, 0.02 m]`. These are physical
    units, which is the env's action space; actions are clipped to the box.
- Seeding: Python `random`, NumPy, PyTorch, SB3 `set_random_seed`, the env
  (`reset(seed)`), and the action/observation spaces in training; seeded
  action space for the random baseline; explicit env seeds for every
  evaluation episode.
- `README.md` and `METHODS_NOTES.md`: new protocol, exact commands, npz
  keys, the statistics definition, the rule-based baseline, a statement that
  both training and held-out evaluation returns are reported, and an empty
  (TBD) results table. The old committed section is kept.
- `.gitignore`: ignore `results/models_*/`.

## Bugs / inaccuracies found

1. **Random baseline was not reproducible.** `env.action_space.sample()` was
   used without seeding the action space. It is now seeded (legacy: seed 888).
2. **Documentation error about TD3 noise.** METHODS_NOTES said sigma=0.1 was
   "the same absolute value for power and waist". SB3 adds the noise in the
   rescaled [-1, 1] action space, so it is 5% of each half-range
   (0.6 dBm and 0.00125 m). The code was unchanged; the note was corrected.
3. **Re-running overwrote the committed data.** The old script always wrote
   `results/multi_algo_data.npz` (this is how the 400-episode array was lost).
   New runs now write to separate files.
4. `plt.show()` blocked headless or batch runs. It is now opt-in (`--show`).
5. Noted, not changed: the env returns `terminated=True` (not `truncated`)
   at the 80-step time limit. `train_sac.py` and `compare_baseline.py` use
   `max_steps=100`. Both are documented in METHODS_NOTES.md.

## Commands

```bash
pip install -r requirements.txt

# Strong protocol: 3 algos x 10 seeds x 1000 episodes + 50-episode held-out eval, all figures
python experiments/train_multi_seed.py

# Same protocol with 600 training episodes
python experiments/train_multi_seed.py --episodes 600

# Redraw figures/summary from results/multi_algo_strong.npz (SE bands optional)
python experiments/train_multi_seed.py --plot-only --band se

# Old 3-seed / 600-episode protocol (writes *_rerun files)
python experiments/train_multi_seed.py --preset legacy

# Redraw the committed old figure
python experiments/train_multi_seed.py --preset legacy-committed --plot-only

# Quick smoke test (NOT results), written outside results/
python experiments/train_multi_seed.py --episodes 12 --seeds 2 --eval-episodes 3 --window 3 --out /tmp/fso_smoke/smoke
```

Then copy the numbers from `results/multi_algo_strong_summary.md` into the
TBD table in README.md.

## Verification done

- Dependencies from `requirements.txt` (Python 3.13, torch 2.15.0 CPU,
  SB3 2.9.2, Gymnasium 1.4.0) were installed.
- Smoke test (2 seeds, 12 episodes, 3 eval episodes, written to /tmp) ran end
  to end in about 10 s: npz, summary, and all 3 figures. A second identical
  run gave identical arrays. `--plot-only`, `--preset legacy`, and
  `--preset legacy-committed --plot-only` were also exercised.
- Smoke-test numbers are not results and are not included anywhere.
