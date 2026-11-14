from nilm.regional import RegionalConfig, build_regional
from nilm.experiment import run_experiment, PRESETS
from nilm.engine import TrainConfig
cfg = RegionalConfig(train_per_home=1150, eval_per_home=600)
data = build_regional(cfg)
variants = PRESETS["architectures"] + PRESETS["classical"]
order = ["MLP", "CNN1D", "Rule", "LogReg", "GBoost", "CNN-GRU", "LSTM", "GRU"]
by_name = {v["name"]: v for v in variants}
run_experiment("multiregion", cfg, TrainConfig(epochs=5, max_train=0), [by_name[n] for n in order], data=data)
