# Multi-region NILM comparison

Training regions: austin, california, newyork, puertorico. Seed 42.

Each model receives 30 consecutive one-minute whole-home readings (grid plus available solar, converted to watts). Appliance meters provide ON/OFF and watts at the last minute. Unmetered appliance labels are excluded from loss and scoring.

Each participating home's timeline is split 70% / 15% / 15%. In transfer runs, only training regions contribute to fitting and validation. Approximately 20% of eligible homes per region are held out completely. Short gaps of up to five minutes are interpolated within partitions. Windows with missing aggregate values are excluded. The scaler and physical ON thresholds use training windows only. Probability cutoffs for learned classifiers use validation macro-F1. The rule baseline uses a fixed decision cutoff.

Per-home window settings: training 2500, evaluation 600 (0 means all unless a minimum coverage is specified). Windows are sampled evenly over each eligible period. Power errors are measured at sampled minutes. The Puerto Rico source has sparse appliance coverage. A regional cell needs metered data and both states to count toward the regional mean.

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
| austin | 20/25 (80%) | 50,000 | 7,256,324 |
| california | 18/23 (78%) | 45,000 | 6,624,967 |
| newyork | 20/25 (80%) | 50,000 | 3,647,869 |
| puertorico | 13/25 (52%) | 32,500 | 1,063,027 |

Home coverage counts participating homes. Window coverage counts sampled minutes within the earliest 70% of those homes' timelines.

## Classes and thresholds

| Appliance | ON above (W) | Training ON / OFF | Unseen ON / OFF |
|---|---:|---:|---:|
| air1 | 37.6 | 21,071 / 101,426 | 1,433 / 6,367 |
| furnace1 | 49.4 | 26,769 / 88,100 | 1,604 / 6,196 |
| drye1 | 42.9 | 2,060 / 77,931 | 79 / 2,921 |
| refrigerator1 | 18.4 | 73,063 / 46,512 | 5,111 / 3,289 |
| lights_plugs1 | 11.4 | 30,048 / 32,407 | 1,565 / 232 |
| waterheater1 | 31.9 | 4,735 / 29,968 | 39 / 2,361 |

## Results on completely unseen homes

The minimum course benchmarks per appliance are 85% accuracy, 75% balanced accuracy and 0.70 macro-F1. Higher targets are 90%, 85% and 0.80 respectively.

| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | MAE (W) | RMSE (W) | Benchmark |
|---|---|---:|---:|---:|---:|---:|---|
| MLP | air1 | 89.2% | 73.4% | 0.780 | 124.1 | 395.9 | Below |
| MLP | furnace1 | 87.4% | 72.7% | 0.767 | 75.4 | 174.8 | Below |
| MLP | drye1 | 98.6% | 80.8% | 0.849 | 56.0 | 445.5 | Minimum |
| MLP | refrigerator1 | 71.1% | 66.3% | 0.666 | 45.4 | 69.5 | Below |
| MLP | lights_plugs1 | 67.2% | 70.5% | 0.575 | 76.5 | 108.0 | Below |
| MLP | waterheater1 | 89.5% | 90.9% | 0.583 | 69.1 | 564.4 | Below |
| TCN | air1 | 88.9% | 72.2% | 0.769 | 124.5 | 392.8 | Below |
| TCN | furnace1 | 87.9% | 73.1% | 0.773 | 73.4 | 170.6 | Below |
| TCN | drye1 | 99.1% | 88.4% | 0.904 | 61.9 | 477.8 | Excellent |
| TCN | refrigerator1 | 69.2% | 64.7% | 0.649 | 47.9 | 72.2 | Below |
| TCN | lights_plugs1 | 59.9% | 67.6% | 0.524 | 76.5 | 107.9 | Below |
| TCN | waterheater1 | 88.0% | 80.0% | 0.549 | 64.7 | 455.6 | Below |
| Transformer | air1 | 89.2% | 73.1% | 0.778 | 122.9 | 372.0 | Below |
| Transformer | furnace1 | 87.5% | 72.8% | 0.769 | 74.4 | 171.7 | Below |
| Transformer | drye1 | 98.7% | 81.5% | 0.856 | 53.3 | 420.0 | Minimum |
| Transformer | refrigerator1 | 71.0% | 66.4% | 0.668 | 45.1 | 69.6 | Below |
| Transformer | lights_plugs1 | 63.2% | 68.4% | 0.545 | 77.0 | 108.6 | Below |
| Transformer | waterheater1 | 88.2% | 91.5% | 0.571 | 62.6 | 533.3 | Below |
| Inception1D | air1 | 88.9% | 72.8% | 0.773 | 122.6 | 373.2 | Below |
| Inception1D | furnace1 | 87.5% | 73.0% | 0.770 | 75.2 | 173.8 | Below |
| Inception1D | drye1 | 99.1% | 89.7% | 0.909 | 46.4 | 385.2 | Excellent |
| Inception1D | refrigerator1 | 69.3% | 64.5% | 0.647 | 47.5 | 71.7 | Below |
| Inception1D | lights_plugs1 | 72.7% | 61.9% | 0.570 | 74.2 | 105.7 | Below |
| Inception1D | waterheater1 | 88.6% | 90.4% | 0.574 | 37.1 | 269.1 | Below |
| Rule | air1 | 82.6% | 68.9% | 0.697 | 314.2 | 721.2 | Below |
| Rule | furnace1 | 80.3% | 70.0% | 0.700 | 109.4 | 218.0 | Below |
| Rule | drye1 | 81.7% | 72.1% | 0.525 | 425.8 | 976.7 | Below |
| Rule | refrigerator1 | 67.7% | 67.0% | 0.666 | 49.2 | 78.7 | Below |
| Rule | lights_plugs1 | 67.9% | 69.3% | 0.575 | 57.5 | 91.7 | Below |
| Rule | waterheater1 | 84.3% | 89.5% | 0.539 | 244.3 | 724.5 | Below |
| LogReg | air1 | 89.3% | 74.3% | 0.786 | 154.0 | 400.3 | Below |
| LogReg | furnace1 | 87.8% | 73.9% | 0.779 | 75.8 | 173.5 | Below |
| LogReg | drye1 | 97.6% | 80.3% | 0.780 | 65.9 | 448.8 | Minimum |
| LogReg | refrigerator1 | 63.2% | 59.3% | 0.593 | 53.7 | 75.6 | Below |
| LogReg | lights_plugs1 | 65.0% | 37.7% | 0.397 | 64.9 | 100.4 | Below |
| LogReg | waterheater1 | 89.6% | 93.5% | 0.589 | 57.9 | 363.6 | Below |
| GBoost | air1 | 89.1% | 72.3% | 0.771 | 117.4 | 366.7 | Below |
| GBoost | furnace1 | 87.4% | 74.4% | 0.778 | 77.6 | 210.5 | Below |
| GBoost | drye1 | 98.6% | 83.3% | 0.857 | 43.5 | 337.8 | Minimum |
| GBoost | refrigerator1 | 74.2% | 71.3% | 0.719 | 43.2 | 69.1 | Below |
| GBoost | lights_plugs1 | 42.4% | 34.4% | 0.337 | 74.6 | 104.3 | Below |
| GBoost | waterheater1 | 92.6% | 76.1% | 0.584 | 57.5 | 563.1 | Below |
| CNN1D | air1 | 89.1% | 72.9% | 0.776 | 124.7 | 369.2 | Below |
| CNN1D | furnace1 | 86.7% | 73.8% | 0.767 | 75.6 | 170.9 | Below |
| CNN1D | drye1 | 99.1% | 86.6% | 0.903 | 51.1 | 406.6 | Excellent |
| CNN1D | refrigerator1 | 72.0% | 68.5% | 0.690 | 44.3 | 69.9 | Below |
| CNN1D | lights_plugs1 | 49.5% | 59.6% | 0.443 | 77.6 | 108.8 | Below |
| CNN1D | waterheater1 | 92.2% | 89.8% | 0.613 | 47.2 | 453.8 | Below |
| CNN-GRU | air1 | 89.2% | 73.5% | 0.780 | 116.5 | 368.9 | Below |
| CNN-GRU | furnace1 | 86.6% | 74.8% | 0.773 | 73.6 | 168.2 | Below |
| CNN-GRU | drye1 | 99.0% | 89.6% | 0.901 | 50.3 | 397.0 | Excellent |
| CNN-GRU | refrigerator1 | 69.7% | 65.7% | 0.661 | 46.4 | 70.6 | Below |
| CNN-GRU | lights_plugs1 | 36.0% | 60.3% | 0.351 | 82.1 | 113.1 | Below |
| CNN-GRU | waterheater1 | 88.5% | 87.9% | 0.568 | 43.8 | 409.4 | Below |
| LSTM | air1 | 88.9% | 72.4% | 0.771 | 121.3 | 382.6 | Below |
| LSTM | furnace1 | 87.1% | 74.8% | 0.777 | 76.7 | 173.1 | Below |
| LSTM | drye1 | 99.1% | 86.0% | 0.902 | 46.5 | 380.5 | Excellent |
| LSTM | refrigerator1 | 69.0% | 63.7% | 0.637 | 47.9 | 73.0 | Below |
| LSTM | lights_plugs1 | 37.9% | 58.1% | 0.363 | 80.9 | 111.5 | Below |
| LSTM | waterheater1 | 88.7% | 88.0% | 0.570 | 71.6 | 570.5 | Below |
| GRU | air1 | 89.2% | 72.5% | 0.773 | 111.6 | 371.0 | Below |
| GRU | furnace1 | 86.9% | 71.9% | 0.758 | 74.3 | 172.6 | Below |
| GRU | drye1 | 99.1% | 91.5% | 0.913 | 41.4 | 344.4 | Excellent |
| GRU | refrigerator1 | 69.4% | 65.0% | 0.653 | 46.8 | 71.8 | Below |
| GRU | lights_plugs1 | 38.7% | 55.2% | 0.365 | 82.2 | 113.0 | Below |
| GRU | waterheater1 | 90.1% | 86.2% | 0.580 | 49.1 | 427.6 | Below |
| CNN1D-noweight | air1 | 89.4% | 74.0% | 0.786 | 115.2 | 355.1 | Below |
| CNN1D-noweight | furnace1 | 86.6% | 74.5% | 0.771 | 76.8 | 172.1 | Below |
| CNN1D-noweight | drye1 | 99.0% | 85.3% | 0.889 | 46.5 | 385.3 | Excellent |
| CNN1D-noweight | refrigerator1 | 72.3% | 68.0% | 0.685 | 44.0 | 69.6 | Below |
| CNN1D-noweight | lights_plugs1 | 55.3% | 62.0% | 0.482 | 74.9 | 106.8 | Below |
| CNN1D-noweight | waterheater1 | 91.0% | 80.3% | 0.576 | 56.4 | 556.9 | Below |
| CNN1D-thr0.5 | air1 | 87.7% | 81.4% | 0.803 | 124.3 | 361.6 | Minimum |
| CNN1D-thr0.5 | furnace1 | 82.4% | 82.9% | 0.772 | 77.0 | 163.5 | Below |
| CNN1D-thr0.5 | drye1 | 97.1% | 96.7% | 0.810 | 52.0 | 402.8 | Excellent |
| CNN1D-thr0.5 | refrigerator1 | 71.9% | 68.8% | 0.693 | 44.3 | 70.3 | Below |
| CNN1D-thr0.5 | lights_plugs1 | 44.4% | 57.4% | 0.406 | 78.5 | 109.5 | Below |
| CNN1D-thr0.5 | waterheater1 | 72.9% | 85.0% | 0.472 | 57.6 | 454.9 | Below |
| CNN1D-focal | air1 | 89.3% | 73.3% | 0.780 | 111.7 | 350.0 | Below |
| CNN1D-focal | furnace1 | 86.7% | 74.7% | 0.773 | 75.1 | 168.6 | Below |
| CNN1D-focal | drye1 | 99.2% | 89.1% | 0.917 | 43.9 | 372.4 | Excellent |
| CNN1D-focal | refrigerator1 | 71.7% | 67.9% | 0.685 | 44.8 | 70.4 | Below |
| CNN1D-focal | lights_plugs1 | 48.6% | 60.2% | 0.439 | 74.9 | 106.2 | Below |
| CNN1D-focal | waterheater1 | 90.6% | 88.9% | 0.591 | 44.5 | 316.3 | Below |

## Performance by region

Macro-F1 averaged equally across appliances with both ON and OFF support in each region.

| Model | austin | california | newyork | puertorico | Training seconds | Inference ms per 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.786 | 0.850 | 0.684 | 0.485 | 24.9 | 10.4 |
| TCN | 0.796 | 0.840 | 0.672 | 0.484 | 34.1 | 22.8 |
| Transformer | 0.801 | 0.852 | 0.666 | 0.480 | 44.8 | 14.1 |
| Inception1D | 0.799 | 0.836 | 0.680 | 0.495 | 15.7 | 14.2 |
| Rule | 0.666 | 0.693 | 0.611 | 0.451 | 2.5 | 1.6 |
| LogReg | 0.773 | 0.808 | 0.600 | 0.476 | 4.5 | 7.4 |
| GBoost | 0.823 | 0.876 | 0.632 | 0.450 | 74.6 | 72.1 |
| CNN1D | 0.826 | 0.849 | 0.652 | 0.462 | 127.0 | 55.9 |
| CNN-GRU | 0.811 | 0.833 | 0.651 | 0.476 | 75.3 | 57.9 |
| LSTM | 0.797 | 0.829 | 0.651 | 0.470 | 215.2 | 76.9 |
| GRU | 0.803 | 0.850 | 0.643 | 0.488 | 304.4 | 80.0 |
| CNN1D-noweight | 0.820 | 0.852 | 0.693 | 0.461 | 127.0 | 48.3 |
| CNN1D-thr0.5 | 0.797 | 0.733 | 0.624 | 0.526 | 117.9 | 50.4 |
| CNN1D-focal | 0.825 | 0.846 | 0.664 | 0.468 | 116.0 | 67.3 |

## Model trade-offs

Inception1D had the highest observed mean unseen-home macro-F1 (0.707). Inception1D had the lowest mean unseen-home MAE (67.2 W). The table averages appliances equally. Results come from one seeded home split.

| Model | Future-time macro-F1 | Unseen macro-F1 | Unseen MAE (W) | Appliances meeting minimum | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| Inception1D | 0.630 | 0.707 | 67.2 | 1 | 10,760 |
| MLP | 0.634 | 0.703 | 74.4 | 1 | 66,572 |
| CNN1D | 0.664 | 0.699 | 70.1 | 1 | 122,668 |
| Transformer | 0.637 | 0.698 | 72.5 | 1 | 18,220 |
| TCN | 0.627 | 0.695 | 74.8 | 1 | 8,188 |
| GBoost | 0.694 | 0.674 | 69.0 | 1 | 0 |
| GRU | 0.649 | 0.674 | 67.6 | 1 | 49,916 |
| CNN-GRU | 0.657 | 0.672 | 68.8 | 1 | 33,548 |
| LSTM | 0.659 | 0.670 | 74.2 | 1 | 63,484 |
| LogReg | 0.589 | 0.654 | 78.7 | 1 | 0 |
| Rule | 0.615 | 0.617 | 200.1 | 0 | 0 |

### Original four appliance targets

The overall ranking includes lighting and water heating. This table shows performance on the four original appliance targets within the expanded experiment.

| Model | Original-target macro-F1 | Original-target MAE (W) |
|---|---:|---:|
| MLP | 0.765 | 75.2 |
| TCN | 0.774 | 76.9 |
| Transformer | 0.768 | 73.9 |
| Inception1D | 0.775 | 72.9 |
| Rule | 0.647 | 224.7 |
| LogReg | 0.734 | 87.3 |
| GBoost | 0.781 | 70.4 |
| CNN1D | 0.784 | 73.9 |
| CNN-GRU | 0.779 | 71.7 |
| LSTM | 0.772 | 73.1 |
| GRU | 0.774 | 68.5 |

## Effect of training sample size

Both runs used the same homes, target circuits, window length and test labels. The smaller run used 1,150 training windows per home. The expanded run used 2,500.

| Model | 1,150 per home | 2,500 per home | Change in unseen macro-F1 |
|---|---:|---:|---:|
| MLP | 0.704 | 0.703 | -0.000 |
| TCN | 0.704 | 0.695 | -0.010 |
| Transformer | 0.695 | 0.698 | +0.003 |

## Controlled CNN1D comparison

Architecture, inputs, splits, regression objective, seed and training budget are shared.

| Classification and cutoff | Unseen macro-F1 | Unseen balanced accuracy | Unseen MAE (W) |
|---|---:|---:|---:|
| Weighted BCE, cutoff tuned on validation | 0.699 | 0.752 | 70.1 |
| Unweighted BCE, cutoff tuned on validation | 0.698 | 0.740 | 69.0 |
| Weighted BCE, fixed 0.5 cutoff | 0.659 | 0.787 | 72.3 |
| Weighted focal loss, cutoff tuned on validation | 0.697 | 0.757 | 65.8 |

The detailed CSV includes validation, future-time and unseen-home splits, per-region support counts and the four confusion matrix counts. The Streamlit app shows training curves, confusion matrices and actual versus predicted power.

## Reproduce

```powershell
python -m nilm.experiment --preset multiregion --name expanded --epochs 3 --max-train 0 --train-per-home 2500 --eval-per-home 600 --regions austin california newyork puertorico --window 30 --appliances air1 furnace1 drye1 refrigerator1 lights_plugs1 waterheater1 --only MLP,TCN,Transformer,Inception1D,Rule,LogReg,GBoost,CNN1D,CNN-GRU,LSTM,GRU,CNN1D-noweight,CNN1D-thr0.5,CNN1D-focal --techniques
python -m nilm.traces expanded
python -m nilm.dashboard expanded
streamlit run app.py
```
