"""Build timestamped actual-versus-predicted examples from unseen homes."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .classical import predict_classical
from .config import RESULTS_DIR, region_cache
from .engine import predict
from .models import MODELS, build
from .regional import interpolate_short_gaps, read_home


def make_traces(name):
    directory = RESULTS_DIR / name
    blob = json.loads((directory / "results.json").read_text())
    meta, results = blob["meta"], blob["results"]
    apps, width = meta["data_cfg"]["appliances"], meta["data_cfg"]["window"]
    scale = np.asarray([meta["p_scale"][a] for a in apps], dtype=np.float32)
    rows = []
    for region, homes in meta["regional_homes"].items():
        candidates = []
        for hid in homes["unseen"]:
            home = read_home(region_cache(region) / f"{hid}.parquet", apps)
            # Prefer homes with more measured targets in the example week.
            part = home.iloc[:7 * 1440]
            targets = part[apps].to_numpy()
            observed = np.isfinite(targets).sum(axis=0)
            candidates.append(((observed > 0).sum(), observed.sum(), hid, part))
        _, _, hid, part = max(candidates, key=lambda candidate: candidate[:3])
        frame = interpolate_short_gaps(part.copy())
        load = frame["aggregate"].to_numpy(np.float32)
        history = sliding_window_view(load, width)
        valid = np.isfinite(history).all(axis=1)
        positions = np.flatnonzero(valid)
        if not len(positions):
            continue
        start = positions[0]
        stop = min(start + 12 * 60, len(history))
        indices = np.arange(start, stop)
        measured = frame.iloc[indices + width - 1]
        X = (history[indices[valid[indices]]] - meta["x_mean"]) / meta["x_std"]
        for result in results:
            variant, model_name = result["variant"], result["model"]
            if model_name in MODELS and result["logic"] == "multi":
                model = build(model_name, width, len(apps))
                model.load_state_dict(torch.load(directory / f"{variant}_{'_'.join(apps)}.pt",
                                                 map_location="cpu", weights_only=True))
                prob, power = predict(model, X)
                power *= scale
            elif model_name not in MODELS:
                model = joblib.load(directory / f"{variant}.joblib")
                prob, power = predict_classical(model, X, meta)
            else:
                continue
            threshold = np.asarray([result["decision_thresholds"][a] for a in apps])
            gated = np.where(prob > threshold, np.maximum(power, 0), 0)
            watts = np.full((len(indices), len(apps)), np.nan)
            watts[valid[indices]] = gated
            for j, app in enumerate(apps):
                rows.append(pd.DataFrame(dict(region=region, home=hid, variant=variant, appliance=app,
                                              time=measured.index, actual_w=measured[app].to_numpy(),
                                              predicted_w=watts[:, j])))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(directory / "prediction_traces.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="multiregion")
    torch.set_num_threads(4)
    make_traces(parser.parse_args().name)
