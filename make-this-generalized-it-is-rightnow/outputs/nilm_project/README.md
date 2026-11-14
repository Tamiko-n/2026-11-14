# ANN-based NILM: CT-466 Complex Computing Problem

This project predicts each appliance's **ON/OFF state** and **power (W)** from **whole-house power only**,
using the Pecan Street 1-minute regional data. It comes with a Streamlit UI (**NILM Studio**) for exploring
the data, training models, comparing them and generating the report.

## Folder layout
```
nilm_project/
├─ app.py                 Streamlit UI (8 tabs)
├─ nilm/
│  ├─ config.py           paths, region file patterns, seed
│  ├─ ingest.py           huge CSV -> per-home parquet cache + appliance coverage survey
│  ├─ data.py             cleaning, train-only thresholds/scaler, time split, leakage-free windows
│  ├─ models.py           MLP, 1D-CNN, LSTM, GRU, CNN-GRU (shared two-head design)
│  ├─ classical.py        non-neural logic: step-threshold rule, LogReg, Gradient Boosting
│  ├─ engine.py           training loop, early stopping, validation threshold tuning, metrics
│  ├─ experiment.py       experiment presets, evaluation on val/test/unseen, saving
│  └─ report.py           auto-generated Markdown report
├─ cache/<region>/        parquet per home + _coverage.csv   (created by ingest)
└─ results/<experiment>/  results.json, metrics.csv, training_log.csv, *.pt, *_preds.npz, REPORT.md
```
The raw regional folders stay where they were supplied (`E:\ann\1minute_data_austin`, ...).
Set `NILM_DATA_ROOT` to the folder containing the regional datasets.

## Legacy single-region experiments
```bash
pip install -r requirements.txt

# 1) one-off: cache the regions you need (streams the CSV, so low RAM is fine)
python -m nilm.ingest --region austin
python -m nilm.ingest --region california      # only needed for the cross-region test

# 2) experiments (seed 42; logs print per epoch and are saved to training_log.csv)
python -m nilm.experiment --preset architectures --name main   # MLP, CNN1D, LSTM, GRU, CNN-GRU
python -m nilm.experiment --preset classical     --name main   # Rule, LogReg, GBoost (appended)
python -m nilm.experiment --preset logic         --name main   # single-task, no class weight, fixed 0.5
python -m nilm.experiment --preset window                      # window 30/120 ablation
python -m nilm.experiment --preset crossregion                 # train Austin -> test California

# 3) report + UI
python -m nilm.report main          # writes results/main/REPORT.md
streamlit run app.py
```

## Design decisions (and why)
| Decision | Reason |
|---|---|
| Input = `grid + solar` | `grid` alone goes negative when panels export; household *consumption* is grid + solar. |
| Derived Δ-channel inside the nets | still computed only from total power, but a 100 W fridge switching is far easier to see as an edge than as a level. |
| 70/15/15 time split **per home** + whole unseen homes | no future information leaks into training; unseen homes test generalisation to new houses. |
| ON threshold = 2-cluster k-means on log(train power) | the threshold adapts to each appliance (fridge about 20-50 W, AC hundreds of W) without looking at test data. |
| Probability cut-off tuned on validation macro-F1 | handles class imbalance (dryer is ON about 2% of the time) without touching test. |
| Class-weighted BCE + Huber loss, two heads | joint ON/OFF + power estimation; Huber is robust to power spikes. |
| Power output gated by predicted state | an appliance predicted OFF reports 0 W, which lowers MAE. |

Generalisation: everything (region, homes, appliances, dates, window, models) is a parameter, both in
`DataConfig` and in the UI's **Train** tab. Any metered circuit in any of the four regions can be targeted.

## Pooled regional training

The default experiment trains five neural architectures and three classical baselines
on Austin, California, New York and Puerto Rico. Each eligible home contributes
chronological windows. A seeded holdout reserves entire homes in each region.
Missing appliance targets are ignored in learning and scoring.

From the project directory, run these PowerShell commands:

```powershell
python -m pip install -r requirements.txt
$env:NILM_DATA_ROOT = "E:/ann"
$env:NILM_CACHE_DIR = "$PWD/cache"
python -m nilm.ingest --region all
python -m nilm.experiment --preset multiregion --epochs 5 --max-train 0
python -m nilm.traces multiregion
streamlit run app.py
```

Existing Austin and California caches under the original project can be reused.
When those caches exist, prepare only New York and Puerto Rico using
`--region newyork` and `--region puertorico`. Add
`--targets air1 furnace1 drye1 refrigerator1` if you only need the four studied
circuits in the new caches. The supplied run used 1,150 evenly
spaced training windows and 600 evaluation windows per eligible home and split.
Set `--train-per-home 0 --eval-per-home 0 --max-train 0` to use all valid windows,
subject to available RAM and training time. Use `--regions` to select a different
pool and `--train-regions` to restrict model fitting for a transfer study.

Open **Results** for benchmark tables, confusion matrices, training curves and
unseen-home power traces. Open **Generalisation** for region-by-appliance metrics.
The CSV and Markdown files for each experiment are under `results/`.
The supplied results and interactive HTML comparison open without the raw datasets.

To reproduce the Austin-only control and score both training pools against the
same unseen-home labels:

```powershell
python -m nilm.experiment --train-regions austin --name austin_transfer --only MLP,CNN1D,GBoost --epochs 5 --max-train 0
python -m nilm.transfer
python -m nilm.report multiregion
python -m nilm.dashboard
```

## Expanded experiment

The expanded benchmark uses eight ANN architectures (MLP, CNN1D, LSTM, GRU,
CNN-GRU, TCN, Transformer and Inception1D), three classical baselines and three
CNN1D variants for isolating class weights, validation cutoff tuning and focal
loss. Every architecture receives the same chronologically sampled data and
the same six output circuits. Lighting and plug loads are evaluated as the
metered circuit recorded in the source data.

From the project directory, with the regional paths configured as above:

```powershell
python -m nilm.ingest --region all --targets air1 furnace1 drye1 refrigerator1 lights_plugs1 waterheater1
python -m nilm.experiment --name expanded --epochs 3 --window 30 --techniques --train-per-home 2500 --eval-per-home 600 --appliances air1 furnace1 drye1 refrigerator1 lights_plugs1 waterheater1
python -m nilm.traces expanded
python -m nilm.dashboard expanded
streamlit run app.py
```

The 50% coverage requirement in this run is measured by source household count
in each region. Training sampled 2,500 eligible minutes per participating home
from the earliest 70% of its timeline. The coverage table in the report shows
both the home counts and the fraction of available training windows sampled.
To require half of each participating home's valid training windows, use
`--minimum-window-coverage 0.5`. This can demand several gigabytes of RAM.
