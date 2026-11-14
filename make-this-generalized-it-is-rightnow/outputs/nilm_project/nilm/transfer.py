"""Compare Austin-only and pooled fitting against a common unseen-home target."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch

from .classical import predict_classical
from .config import RESULTS_DIR
from .engine import grade, metrics, predict
from .models import MODELS, build
from .regional import RegionalConfig, build_regional


def compare_training_regions(pooled="multiregion", baseline="austin_transfer", filename="austin_vs_pooled.csv"):
    blobs = [json.loads((RESULTS_DIR / name / "results.json").read_text())
             for name in (pooled, baseline)]
    settings = dict(blobs[0]["meta"]["data_cfg"])
    if "appliances_minimum_coverage" in settings:
        settings["minimum_window_coverage"] = settings.pop("appliances_minimum_coverage")
    config = RegionalConfig(**settings)
    for key in ("regions", "appliances", "window", "eval_per_home", "seed"):
        if blobs[0]["meta"]["data_cfg"][key] != blobs[1]["meta"]["data_cfg"][key]:
            raise ValueError(f"Comparison requires a shared {key}")
    data = build_regional(config)
    unseen = data["splits"]["unseen"]
    shared = set.intersection(*(set(r["variant"] for r in b["results"]) for b in blobs))
    rows = []
    for name, blob in zip((pooled, baseline), blobs):
        meta = blob["meta"]
        if any(meta["regional_homes"][region]["unseen"] != homes["unseen"]
               for region, homes in data["regional_homes"].items()):
            raise ValueError("The runs must share precisely the same unseen homes")
        X = unseen["X"] * (data["x_std"] / meta["x_std"]) + (data["x_mean"] - meta["x_mean"]) / meta["x_std"]
        for result in blob["results"]:
            variant = result["variant"]
            if variant not in shared:
                continue
            directory = RESULTS_DIR / name
            apps = config.appliances
            if result["model"] in MODELS:
                model = build(result["model"], config.window, len(apps))
                model.load_state_dict(torch.load(directory / f"{variant}_{'_'.join(apps)}.pt",
                                                 map_location="cpu", weights_only=True))
                prob, watts = predict(model, X)
                watts *= np.asarray([meta["p_scale"][a] for a in apps], dtype=np.float32)
            else:
                model = joblib.load(directory / f"{variant}.joblib")
                prob, watts = predict_classical(model, X, meta)
            for region in ["pooled", *config.regions]:
                where = np.ones(len(X), dtype=bool) if region == "pooled" else unseen["R"] == region
                for j, appliance in enumerate(apps):
                    valid = where & np.isfinite(unseen["Y"][:, j])
                    if not valid.any():
                        continue
                    m = metrics(unseen["Y"][valid, j], prob[valid, j], unseen["P"][valid, j],
                                watts[valid, j], result["decision_thresholds"][appliance])
                    m["grade"] = grade(m) if m["n_on"] and m["n_off"] else "Insufficient classes"
                    rows.append(dict(experiment=name, variant=variant, region=region, appliance=appliance,
                                     **{k: v for k, v in m.items() if k != "cm"}))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / pooled / filename, index=False)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--larger", default="multiregion")
    parser.add_argument("--control", default="austin_transfer")
    parser.add_argument("--output", default="austin_vs_pooled.csv")
    args = parser.parse_args()
    torch.set_num_threads(3)
    compare_training_regions(args.larger, args.control, args.output)
