from nilm.regional import RegionalConfig, build_regional
from nilm.experiment import run_experiment, PRESETS
from nilm.engine import TrainConfig
from nilm.report import make_report
from nilm.config import RESULTS_DIR
import torch
torch.set_num_threads(4)
cfg = RegionalConfig(window=30, train_per_home=2500, eval_per_home=600,
    appliances=["air1","furnace1","drye1","refrigerator1","lights_plugs1","waterheater1"])
data = build_regional(cfg)
print(data["coverage_by_region"], flush=True)
for region,h in data["regional_homes"].items():
    assert 2 * len(h["train"]) >= data["coverage_by_region"][region]["raw_homes"], (region,h)
variants = PRESETS["architectures"] + PRESETS["classical"] + PRESETS["logic"]
by_name = {v["name"]:v for v in variants}
order = ["MLP","TCN","Transformer","Inception1D","Rule","LogReg","GBoost","CNN1D","CNN-GRU","LSTM","GRU",
         "CNN1D-noweight","CNN1D-thr0.5","CNN1D-focal"]
run_experiment("expanded", cfg, TrainConfig(epochs=3, max_train=0), [by_name[n] for n in order], data=data)
(RESULTS_DIR/"expanded"/"REPORT.md").write_text(make_report("expanded"),encoding="utf-8")
