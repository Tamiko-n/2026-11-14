"""Step 8-9 - orchestrate experiments, evaluate on held-out data and save everything.

An *experiment* = one DataConfig + a list of *variants*.  A variant is a model
plus the logic it is trained with (multi-task / single-task, class weighting,
threshold tuning, window length ...).  Each variant is evaluated on:
  val     - used only for early stopping + decision thresholds
  test    - last 15 % (in time) of the seen homes        (temporal generalisation)
  unseen  - complete homes never seen in training         (cross-home generalisation)

Usage:
  python -m nilm.experiment --preset architectures
  python -m nilm.experiment --preset all
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from . import models as M
from .classical import run_classical, predict_classical
from .config import REGIONS, RESULTS_DIR
from .data import DataConfig, build_dataset
from .engine import TrainConfig, train_torch, predict, metrics, grade

NEURAL = set(M.MODELS)


def evaluate_split(data, split, prob, pw_scaled, thr, app_idx, scale, select=slice(None)):
    s = data["splits"][split]
    A = data["appliances"]
    out = {}
    for k, j in enumerate(app_idx):
        pw = pw_scaled[:, k] * scale[k]
        labels = s["Y"][select, j]
        valid = np.isfinite(labels)
        if not valid.any():
            continue
        m = metrics(labels[valid], prob[select, k][valid], s["P"][select, j][valid], pw[select][valid], thr[k])
        m["grade"] = grade(m) if m["n_on"] and m["n_off"] else "Insufficient classes"
        out[A[j]] = m
    return out


def run_variant(data, v, tcfg, run_dir, log=print, progress=None):
    """v = {'name':..., 'model':..., 'logic': 'multi'|'single', overrides...}"""
    A = data["appliances"]
    tc = replace(tcfg, **{k: v[k] for k in ("class_weighting", "tune_threshold", "epochs", "focal_gamma") if k in v})
    res = {"variant": v["name"], "model": v["model"], "logic": v.get("logic", "multi"),
           "train_cfg": tc.to_dict(), "splits": {}, "history": []}
    groups = [[j] for j in range(len(A))] if v.get("logic") == "single" else [list(range(len(A)))]
    preds = {}
    t_total, params = 0.0, 0
    for g in groups:
        scale = np.array([data["p_scale"][A[j]] for j in g], np.float32)
        if v["model"] in NEURAL:
            model, hist, thr, t = train_torch(v["model"], data, tc, g, log=log, progress=progress)
            params += M.n_params(model)
            res["history"] += [dict(h, appliances="+".join(A[j] for j in g)) for h in hist]
            pf = lambda X: predict(model, X)
            torch.save(model.state_dict(), run_dir / f"{v['name']}_{'_'.join(A[j] for j in g)}.pt")
        else:
            model, thr, t = run_classical(v["model"], data, seed=tc.seed, max_train=tc.max_train, log=log)
            joblib.dump(model, run_dir / f"{v['name']}.joblib")
            sc1 = 1.0 / scale  # classical predicts watts -> rescale to "scaled" convention
            pf = lambda X, model=model, g=g, sc1=sc1: (lambda pr: (pr[0][:, g], pr[1][:, g] * sc1))(
                predict_classical(model, X, data))
            thr = thr[g]
        t_total += t
        for split in ("val", "test", "unseen"):
            if split not in data["splits"]:
                continue
            t1 = time.time()
            prob, pw = pf(data["splits"][split]["X"])
            infer = time.time() - t1
            r = evaluate_split(data, split, prob, pw, thr, g, scale)
            res["splits"].setdefault(split, {}).update(r)
            if "R" in data["splits"][split]:
                regions = data["splits"][split]["R"]
                by_region = res.setdefault("regional_splits", {}).setdefault(split, {})
                for region in np.unique(regions):
                    scores = evaluate_split(data, split, prob, pw, thr, g, scale, regions == region)
                    by_region.setdefault(region, {}).update(scores)
            latency = res.setdefault("infer_ms_per_1k", {})
            latency[split] = latency.get(split, 0.0) + infer / len(prob) * 1e6
            preds.setdefault(split, {})
            for k, j in enumerate(g):
                preds[split][A[j]] = (prob[:, k], np.where(prob[:, k] > thr[k], np.maximum(pw[:, k] * scale[k], 0), 0))
        res.setdefault("decision_thresholds", {}).update({A[j]: float(thr[k]) for k, j in enumerate(g)})
    res["train_time_s"] = t_total
    res["params"] = params
    # save a compact prediction trace for plots (first 3 days of each split)
    np.savez_compressed(run_dir / f"{v['name']}_preds.npz", **{
        f"{split}|{a}|{kind}": arr[:4320].astype(np.float32)
        for split, d in preds.items() for a, (p, w) in d.items()
        for kind, arr in (("prob", p), ("power", w))})
    return res


PRESETS = {
    # 1. architecture comparison: same data, same logic
    "architectures": [{"name": n, "model": n} for n in M.MODELS],
    # 2. ANN vs non-neural logic
    "classical": [{"name": n, "model": n} for n in ["Rule", "LogReg", "GBoost"]],
    # 3. learning-logic ablations on the advanced model
    "logic": [
        {"name": "CNN1D", "model": "CNN1D"},
        {"name": "MLP", "model": "MLP"},
        {"name": "CNN1D-single", "model": "CNN1D", "logic": "single"},
        {"name": "CNN1D-noweight", "model": "CNN1D", "class_weighting": False},
        {"name": "CNN1D-thr0.5", "model": "CNN1D", "tune_threshold": False},
        {"name": "CNN1D-focal", "model": "CNN1D", "focal_gamma": 2.0},
        {"name": "MLP-single", "model": "MLP", "logic": "single"},
    ],
}

REGIONAL_TECHNIQUES = {"CNN1D-noweight", "CNN1D-thr0.5", "CNN1D-focal"}


def run_experiment(exp_name, dcfg: DataConfig, tcfg: TrainConfig, variants, log=print,
                   progress=None, data=None):
    run_dir = RESULTS_DIR / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if data is None:
        data = build_dataset(dcfg, log=log)
    meta = {"experiment": exp_name, "data_cfg": dcfg.to_dict(), "train_cfg": tcfg.to_dict(),
            "thresholds_w": data["thresholds"], "class_counts": data["class_counts"],
            "cleaning": {str(k): v for k, v in data["cleaning"].items()},
            "split_bounds": {str(k): v for k, v in data["split_bounds"].items()},
            "x_mean": data["x_mean"], "x_std": data["x_std"], "p_scale": data["p_scale"],
            "torch": torch.__version__, "created": time.strftime("%Y-%m-%d %H:%M")}
    for key in ("regional_homes", "train_regions", "coverage_by_region"):
        if key in data:
            meta[key] = data[key]
    np.savez_compressed(run_dir / "_truth.npz", **{
        f"{s}|{a}": d["P"][:4320, j] for s, d in data["splits"].items() for j, a in enumerate(data["appliances"])})
    results = []
    prev = load_results(exp_name)
    compatible = ("train_cfg", "train_regions", "regional_homes", "split_bounds",
                  "class_counts", "thresholds_w", "x_mean", "x_std", "p_scale")
    def same_config(previous):
        old_data = dict(previous.get("data_cfg", {}))
        new_data = dict(meta["data_cfg"])
        old_data.setdefault("minimum_window_coverage", old_data.pop("appliances_minimum_coverage", 0.0))
        new_data.setdefault("minimum_window_coverage", 0.0)
        return old_data == new_data and all(previous.get(key) == meta.get(key) for key in compatible)

    if prev and same_config(prev["meta"]):
        names = {v["name"] for v in variants}
        results = [r for r in prev["results"] if r["variant"] not in names]
        log(f"appending to existing experiment ({len(results)} previous variants kept)")
    for i, v in enumerate(variants):
        log(f"=== variant {i + 1}/{len(variants)}: {v['name']}")
        sub = (lambda f, i=i: progress((i + f) / len(variants))) if progress else None
        r = run_variant(data, v, tcfg, run_dir, log=log, progress=sub)
        results.append(r)
        _save(run_dir, meta, results)
        if progress:
            progress((i + 1) / len(variants))
    meta["total_time_s"] = time.time() - t0
    _save(run_dir, meta, results)
    log(f"saved -> {run_dir}")
    return meta, results


def _save(run_dir, meta, results):
    manifest = run_dir / "results.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=str))
    temporary.replace(manifest)
    rows = []
    for r in results:
        for split, d in r["splits"].items():
            for a, m in d.items():
                rows.append({"variant": r["variant"], "model": r["model"], "logic": r["logic"],
                             "split": split, "appliance": a,
                             **{k: m[k] for k in m if k != "cm"}, "params": r["params"],
                             "train_time_s": r["train_time_s"]})
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    regional_rows = []
    for r in results:
        for split, regions in r.get("regional_splits", {}).items():
            for region, appliances in regions.items():
                for appliance, m in appliances.items():
                    (tn, fp), (fn, tp) = m["cm"]
                    regional_rows.append(dict(variant=r["variant"], split=split, region=region,
                                              appliance=appliance, tn=tn, fp=fp, fn=fn, tp=tp,
                                              **{k: v for k, v in m.items() if k != "cm"}))
    if regional_rows:
        pd.DataFrame(regional_rows).to_csv(run_dir / "regional_metrics.csv", index=False)
    log_rows = [dict(h, variant=r["variant"]) for r in results for h in r["history"]]
    pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)


def load_results(exp_name):
    p = RESULTS_DIR / exp_name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_experiments():
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "results.json").exists())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="multiregion",
                    choices=list(PRESETS) + ["all", "window", "crossregion", "multiregion"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=0,
                    help="optional shared training cap; 0 uses the assembled training windows")
    ap.add_argument("--only", default="", help="comma list: run only these variant names of the preset")
    ap.add_argument("--regions", nargs="+", choices=REGIONS)
    ap.add_argument("--train-regions", nargs="+", help="optional subset for a region transfer experiment")
    ap.add_argument("--appliances", nargs="+", help="metered appliance or circuit columns")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--train-per-home", type=int, default=2500, help="evenly spaced windows per home; 0 uses all")
    ap.add_argument("--minimum-window-coverage", type=float, default=0,
                    help="optionally retain at least this fraction of each home's usable training windows")
    ap.add_argument("--techniques", action="store_true", help="include controlled CNN1D technique ablations")
    ap.add_argument("--eval-per-home", type=int, default=600, help="windows per home per split; 0 uses all")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.epochs < 1 or a.threads < 1 or a.max_train < 0:
        ap.error("epochs and threads must be positive and max-train must be nonnegative")
    torch.set_num_threads(a.threads)
    if a.only:
        keep = set(a.only.split(","))
        PRESETS = {k: [v for v in vs if v["name"] in keep] for k, vs in PRESETS.items()}
    dcfg, tcfg = DataConfig(), TrainConfig(epochs=a.epochs, max_train=a.max_train)
    if a.preset == "multiregion":
        from .regional import RegionalConfig, build_regional
        dcfg = RegionalConfig(window=a.window, train_per_home=a.train_per_home, eval_per_home=a.eval_per_home,
                              minimum_window_coverage=a.minimum_window_coverage,
                              **({"appliances": a.appliances} if a.appliances else {}),
                              **({"regions": list(dict.fromkeys(a.regions))} if a.regions else {}))
        variants = PRESETS["architectures"] + PRESETS["classical"]
        if a.techniques:
            variants += [v for v in PRESETS["logic"] if v["name"] in REGIONAL_TECHNIQUES]
        if not variants:
            ap.error("no matching model names")
        name = a.name or ("multiregion" if not a.train_regions else "region_transfer")
        data = build_regional(dcfg, train_regions=a.train_regions)
        run_experiment(name, dcfg, tcfg, variants, data=data)
        from .report import make_report
        (RESULTS_DIR / name / "REPORT.md").write_text(make_report(name), encoding="utf-8")
    elif a.preset == "all":
        variants = PRESETS["architectures"] + PRESETS["classical"] + PRESETS["logic"]
        run_experiment(a.name or "main", dcfg, tcfg, variants)
    elif a.preset == "window":   # window-length ablation for the advanced model
        for w in (30, 120):
            run_experiment(f"window_{w}", replace(dcfg, window=w), tcfg,
                           [{"name": f"CNN1D-w{w}", "model": "CNN1D"}])
    elif a.preset == "crossregion":   # train Austin, test on unseen California homes
        dcfg = replace(dcfg, appliances=["air1", "furnace1", "refrigerator1"],
                       homes=[661, 1642, 2335, 2818, 3456, 4767], unseen_homes=[9612],
                       unseen_region="california", unseen_start="2014-05-01", unseen_end="2014-08-31")
        run_experiment(a.name or "crossregion", dcfg, tcfg,
                       [{"name": n, "model": n} for n in ("MLP", "CNN1D")])
    else:
        run_experiment(a.name or a.preset, dcfg, tcfg, PRESETS[a.preset])
