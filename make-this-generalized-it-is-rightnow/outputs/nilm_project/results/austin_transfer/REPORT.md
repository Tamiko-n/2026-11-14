# Multi-region NILM comparison

Training regions: austin. Seed 42.

Each model receives 60 consecutive one-minute whole-home readings (grid plus available solar, converted to watts). Appliance meters provide ON/OFF and watts at the last minute. Unmetered appliance labels are excluded from loss and scoring.

Each participating home's timeline is split 70% / 15% / 15%. In transfer runs, only training regions contribute to fitting and validation. Approximately 20% of eligible homes per region are held out completely. Short gaps of up to five minutes are interpolated within partitions. Windows with missing aggregate values are excluded. The scaler and physical ON thresholds use training windows only. Probability cutoffs for learned classifiers use validation macro-F1. The rule baseline uses a fixed decision cutoff.

Per-home window caps: training 1150, evaluation 600 (0 means all). Windows are sampled evenly over each eligible period. Power errors are measured at sampled minutes.

| Region | Training homes | Completely unseen homes | Excluded homes |
|---|---|---|---|
| austin | 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4767, 5746, 6139, 661, 7536, 7719, 7951, 8156, 9019, 9160, 9278, 9922 | 4373, 7800, 7901, 8386, 8565 |  |
| california |  | 5938, 8342, 8574, 8733, 9836 |  |
| newyork |  | 2096, 3700, 3996, 5058, 5587 |  |
| puertorico |  | 4204, 7667 | 10826, 10887, 1092, 11933, 1286, 2006, 2921, 3835, 4046, 4749, 6056, 6406, 7778, 8235, 9911 |

## Classes and thresholds

| Appliance | ON above (W) | Training ON / OFF | Unseen ON / OFF |
|---|---:|---:|---:|
| air1 | 64.6 | 5,262 / 16,588 | 827 / 6,973 |
| furnace1 | 76.9 | 7,187 / 14,602 | 1,529 / 6,271 |
| drye1 | 52.0 | 326 / 16,920 | 87 / 2,913 |
| refrigerator1 | 24.2 | 14,506 / 5,996 | 5,063 / 3,337 |

## Results on completely unseen homes

The minimum course benchmarks per appliance are 85% accuracy, 75% balanced accuracy and 0.70 macro-F1. Higher targets are 90%, 85% and 0.80 respectively.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | MAE (W) | RMSE (W) | Benchmark |
|---|---|---:|---:|---:|---:|---:|---|
| MLP | air1 | 95.3% | 82.8% | 0.863 | 107.3 | 391.0 | Minimum |
| MLP | furnace1 | 87.1% | 70.0% | 0.742 | 78.8 | 179.5 | Below |
| MLP | drye1 | 98.1% | 78.4% | 0.816 | 67.0 | 503.7 | Minimum |
| MLP | refrigerator1 | 63.7% | 61.0% | 0.612 | 52.3 | 75.6 | Below |
| CNN1D | air1 | 96.0% | 85.9% | 0.886 | 85.1 | 343.6 | Excellent |
| CNN1D | furnace1 | 87.7% | 71.7% | 0.760 | 76.3 | 175.9 | Below |
| CNN1D | drye1 | 98.5% | 75.3% | 0.827 | 54.8 | 426.0 | Minimum |
| CNN1D | refrigerator1 | 68.9% | 65.6% | 0.660 | 48.2 | 72.4 | Below |
| GBoost | air1 | 95.9% | 84.4% | 0.881 | 86.2 | 343.3 | Minimum |
| GBoost | furnace1 | 87.1% | 74.0% | 0.769 | 78.0 | 197.6 | Below |
| GBoost | drye1 | 98.2% | 76.8% | 0.815 | 43.0 | 337.4 | Minimum |
| GBoost | refrigerator1 | 75.2% | 73.0% | 0.734 | 42.3 | 68.1 | Below |

## Performance by region

Macro-F1 averaged equally across appliances with both ON and OFF support in each region.

| Model | austin | california | newyork | puertorico | Training seconds | Inference ms per 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.759 | 0.797 | 0.633 | 0.597 | 4.1 | 4.5 |
| CNN1D | 0.787 | 0.821 | 0.672 | 0.614 | 45.2 | 77.4 |
| GBoost | 0.809 | 0.844 | 0.693 | 0.547 | 15.3 | 35.6 |

The detailed CSV includes validation, future-time and unseen-home splits, per-region support counts and the four confusion matrix counts. The dashboard shows training curves, confusion matrices and actual versus predicted power.

## Reproduce

```powershell
python -m nilm.experiment --preset multiregion --name austin_transfer --epochs 5 --max-train 0 --train-per-home 1150 --eval-per-home 600 --regions austin california newyork puertorico --only MLP,CNN1D,GBoost --train-regions austin
python -m nilm.traces austin_transfer
streamlit run app.py
```
