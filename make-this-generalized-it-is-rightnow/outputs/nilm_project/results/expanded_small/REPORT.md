# Multi-region NILM comparison

Training regions: austin, california, newyork, puertorico. Seed 42.

Each model receives 30 consecutive one-minute whole-home readings (grid plus available solar, converted to watts). Appliance meters provide ON/OFF and watts at the last minute. Unmetered appliance labels are excluded from loss and scoring.

Each participating home's timeline is split 70% / 15% / 15%. In transfer runs, only training regions contribute to fitting and validation. Approximately 20% of eligible homes per region are held out completely. Short gaps of up to five minutes are interpolated within partitions. Windows with missing aggregate values are excluded. The scaler and physical ON thresholds use training windows only. Probability cutoffs for learned classifiers use validation macro-F1. The rule baseline uses a fixed decision cutoff.

Per-home window settings: training 1150, evaluation 600 (0 means all unless a minimum coverage is specified). Windows are sampled evenly over each eligible period. Power errors are measured at sampled minutes. The Puerto Rico source has sparse appliance coverage. A regional cell needs metered data and both states to count toward the regional mean.

| Region | Training homes | Completely unseen homes | Excluded homes |
|---|---|---|---|
| austin | 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4767, 5746, 6139, 661, 7536, 7719, 7951, 8156, 9019, 9160, 9278, 9922 | 4373, 7800, 7901, 8386, 8565 |  |
| california | 1450, 1524, 1731, 203, 2606, 3687, 3864, 3938, 4495, 4934, 6377, 6547, 7062, 7114, 8061, 9213, 9612, 9775 | 5938, 8342, 8574, 8733, 9836 |  |
| newyork | 1222, 1240, 1417, 142, 2318, 2358, 27, 3000, 3488, 3517, 387, 4283, 4550, 558, 5679, 5982, 5997, 9053, 914, 950 | 2096, 3700, 3996, 5058, 5587 |  |
| puertorico | 10826, 11126, 11386, 1245, 2006, 2159, 371, 3987, 6056, 6406, 7667, 8235, 939 | 4204, 8997, 11933 | 10887, 1092, 1286, 2921, 3835, 4046, 4749, 7778, 9911 |

Home-level dates, observed target channels and cleaning counts are in `results.json`.

### Training coverage by region

| Region | Training homes / source homes | Training windows used | Usable training windows available |
|---|---:|---:|---:|
| austin | 20/25 (80%) | 23,000 | 7,256,324 |
| california | 18/23 (78%) | 20,700 | 6,624,967 |
| newyork | 20/25 (80%) | 23,000 | 3,647,869 |
| puertorico | 13/25 (52%) | 14,950 | 1,063,027 |

Home coverage counts participating homes. Window coverage counts sampled minutes within the earliest 70% of those homes' timelines.

## Classes and thresholds

| Appliance | ON above (W) | Training ON / OFF | Unseen ON / OFF |
|---|---:|---:|---:|
| air1 | 37.7 | 9,666 / 46,682 | 1,433 / 6,367 |
| furnace1 | 49.5 | 12,301 / 40,535 | 1,604 / 6,196 |
| drye1 | 41.3 | 988 / 35,806 | 79 / 2,921 |
| refrigerator1 | 18.4 | 33,492 / 21,509 | 5,111 / 3,289 |
| lights_plugs1 | 11.5 | 13,813 / 14,910 | 1,565 / 232 |
| waterheater1 | 31.9 | 2,112 / 13,851 | 39 / 2,361 |

## Results on completely unseen homes

The minimum course benchmarks per appliance are 85% accuracy, 75% balanced accuracy and 0.70 macro-F1. Higher targets are 90%, 85% and 0.80 respectively.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | MAE (W) | RMSE (W) | Benchmark |
|---|---|---:|---:|---:|---:|---:|---|
| MLP | air1 | 89.1% | 73.2% | 0.777 | 125.3 | 390.8 | Below |
| MLP | furnace1 | 87.7% | 73.3% | 0.774 | 75.3 | 174.8 | Below |
| MLP | drye1 | 98.7% | 82.7% | 0.863 | 61.5 | 488.9 | Minimum |
| MLP | refrigerator1 | 69.7% | 64.6% | 0.647 | 47.8 | 70.7 | Below |
| MLP | lights_plugs1 | 73.4% | 67.7% | 0.600 | 77.0 | 109.5 | Below |
| MLP | waterheater1 | 87.5% | 88.6% | 0.561 | 72.2 | 587.6 | Below |
| TCN | air1 | 88.7% | 72.4% | 0.768 | 137.1 | 414.6 | Below |
| TCN | furnace1 | 87.7% | 72.8% | 0.770 | 74.7 | 172.0 | Below |
| TCN | drye1 | 98.8% | 89.5% | 0.886 | 60.2 | 464.4 | Excellent |
| TCN | refrigerator1 | 65.7% | 62.2% | 0.624 | 52.3 | 77.3 | Below |
| TCN | lights_plugs1 | 73.8% | 72.7% | 0.622 | 72.7 | 105.4 | Below |
| TCN | waterheater1 | 88.5% | 81.6% | 0.556 | 62.4 | 477.3 | Below |
| Transformer | air1 | 89.2% | 73.6% | 0.781 | 124.5 | 374.7 | Below |
| Transformer | furnace1 | 87.8% | 74.0% | 0.778 | 74.6 | 170.5 | Below |
| Transformer | drye1 | 98.5% | 82.0% | 0.846 | 60.8 | 474.4 | Minimum |
| Transformer | refrigerator1 | 69.0% | 64.0% | 0.641 | 47.9 | 70.9 | Below |
| Transformer | lights_plugs1 | 64.8% | 67.7% | 0.552 | 78.8 | 110.7 | Below |
| Transformer | waterheater1 | 88.7% | 90.5% | 0.575 | 73.0 | 609.9 | Below |

## Performance by region

Macro-F1 averaged equally across appliances with both ON and OFF support in each region.

| Model | austin | california | newyork | puertorico | Training seconds | Inference ms per 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.785 | 0.834 | 0.692 | 0.488 | 12.8 | 4.4 |
| TCN | 0.788 | 0.810 | 0.678 | 0.484 | 11.8 | 11.9 |
| Transformer | 0.789 | 0.836 | 0.666 | 0.510 | 14.7 | 8.0 |

## Model trade-offs

TCN had the highest observed mean unseen-home macro-F1 (0.704). MLP had the lowest mean unseen-home MAE (76.5 W). The table averages appliances equally. Results come from one seeded home split.

| Model | Future-time macro-F1 | Unseen macro-F1 | Unseen MAE (W) | Appliances meeting minimum | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| TCN | 0.620 | 0.704 | 76.6 | 1 | 8,188 |
| MLP | 0.626 | 0.704 | 76.5 | 1 | 66,572 |
| Transformer | 0.633 | 0.695 | 76.6 | 1 | 18,220 |

The detailed CSV includes validation, future-time and unseen-home splits, per-region support counts and the four confusion matrix counts. The Streamlit app shows training curves, confusion matrices and actual versus predicted power.

## Reproduce

```powershell
python -m nilm.experiment --preset multiregion --name expanded_small --epochs 3 --max-train 0 --train-per-home 1150 --eval-per-home 600 --regions austin california newyork puertorico --window 30 --appliances air1 furnace1 drye1 refrigerator1 lights_plugs1 waterheater1 --only MLP,TCN,Transformer
python -m nilm.traces expanded_small
python -m nilm.dashboard expanded_small
streamlit run app.py
```
