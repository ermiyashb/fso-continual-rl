# Revision experiments (paper v3)

All code for the v3 revision lives in this directory. It does **not** modify the original
simulator (`environment/`, `physics/`) or the original training/evaluation scripts; the
corrected and partially observable protocols are implemented as Gymnasium wrappers in
`common.py`. All outputs are written to `results/revision/`.

## Environment

```
python3.13 -m venv venv && source venv/bin/activate
pip install numpy==2.5.4 matplotlib==3.11.2 gymnasium==1.4.0 stable-baselines3==2.9.2 \
            sb3-contrib==2.9.2 scipy==1.18.1 torch==2.15.0 --extra-index-url https://download.pytorch.org/whl/cpu
```
Versions actually used: CPython 3.13.5, numpy 2.5.4, matplotlib 3.11.2, gymnasium 1.4.0,
stable-baselines3 2.9.2, sb3-contrib 2.9.2, torch 2.15.0+cpu, scipy 1.18.1.
The original-protocol results (`results/multi_algo_strong.npz`) were produced earlier with the
pins in the top-level `requirements.txt`.

Run everything from the repository root.

## Files

| file | purpose |
|---|---|
| `common.py` | constants, seeds, wrappers (`TruncateAtLimit`, `FullObsTime`, `PartialObsDelay`), `make_env`, held-out evaluation |
| `policies.py` | rule (2 cm / 3.5 cm), fixed action, min-power and reward-max oracles, delayed-SNR rule, gain tracking |
| `baselines_nonlearned.py` | grid search of the best constant action and gain-tracking margin on validation episodes; held-out evaluation of all non-learned references |
| `train_revision.py` | TD3/SAC/PPO/RecurrentPPO training with validation-based checkpoint selection |
| `beam_wander_calibration.py` | simulator beam wander vs Andrews-Phillips |
| `adaptation_metrics.py` | power/waist adaptation metrics on all held-out steps |
| `analysis.py` | all statistics (paired t, exact Wilcoxon, Holm, IQM, bootstrap CIs) |
| `figures.py`, `make_tables.py` | figures and LaTeX tables / number macros |

Seeds: training seeds 100-109 (index 0-9); held-out env seeds 1,000,000 + 50*i + j;
checkpoint-selection validation seeds 2,000,000-2,000,019; baseline-tuning validation seeds
3,000,000-3,000,199.

## Exact commands

```
# 1. non-learned references (grid search on validation episodes, then held-out evaluation)
python experiments/revision/baselines_nonlearned.py --workers 6

# 2. full-observation corrected protocol (truncation + time feature, 1000 episodes x 80 steps,
#    validation every 50 episodes on 20 episodes; saves final and best-val checkpoints)
python experiments/revision/train_revision.py --variants full --algos TD3 SAC PPO --seeds 10 --workers 6

# 3. partially observable variants (SNR history only, delay k = 0, 1, 3)
python experiments/revision/train_revision.py --variants po_k0 po_k1 po_k3 --algos TD3 SAC PPO RPPO --seeds 10 --workers 6

#    optional: run only some seed indices (0..9), e.g. to split runs across machines;
#    this only selects jobs, training itself is unchanged. Example:
python experiments/revision/train_revision.py --variants po_k3 --algos SAC TD3 --seed-idx 6 7 8 9

# 4. beam-wander calibration
python experiments/revision/beam_wander_calibration.py

# 5. adaptation metrics (needs the trained full-observation models)
python experiments/revision/adaptation_metrics.py

# 6. statistics, figures, tables
python experiments/revision/analysis.py
python experiments/revision/figures.py
python experiments/revision/make_tables.py results/revision/tables
```
`train_revision.py` skips runs whose result file already exists, so it can be restarted.

## Outputs (`results/revision/`)

- `nonlearned.npz`, `nonlearned_selection.json` - non-learned references and selected grid values
- `runs/{variant}_{algo}_seed{s}.npz` - per run: training returns, validation curve, best-val episode,
  held-out per-episode metrics for final and best-val checkpoints, wall time
- `models/{variant}_{algo}_seed{s}_final.zip`, `..._bestval.zip` - SB3 checkpoints (818 MB; not included
  in the distributed zip because of size)
- `summary.json`, `stats_*.csv` - all summary statistics and tests
- `adaptation_metrics.json`, `beam_wander_calibration.csv`
- `figures/*.png`, `figures_info.json`, `tables/*.tex`

## Machines

The full-observation runs and the `po_k0`/`po_k1` runs were trained on a Linux box. Of the 40
`po_k3` runs, 27 (SAC and TD3 seeds 100-105, PPO seeds 100-107, RPPO seeds 100-106) were trained on
a Windows 11 PC (Intel i3-1220P) with the same code, seeds, hyperparameters, CPython 3.13.5 and
torch 2.15.0+cpu, using `--seed-idx`; the remaining 13 on the Linux box. Evaluation episodes are
identical across machines; learning dynamics are not bit-identical across machines because of
floating-point nondeterminism. Summed per-run wall time over all 150 runs is 152.3 h (runs were
executed 6-8 in parallel, so wall times include CPU contention).
