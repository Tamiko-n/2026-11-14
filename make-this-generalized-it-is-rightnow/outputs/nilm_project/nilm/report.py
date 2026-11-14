"""Generate the short CCP report (Markdown) from a saved experiment."""
import json

import pandas as pd

from .config import RESULTS_DIR
from .engine import MIN_T, HIGH_T
from .models import DESCRIPTIONS


def _fmt(v, pct=False):
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def benchmark_table(results, split="test"):
    rows = []
    for r in results:
        for a, m in r["splits"].get(split, {}).items():
            rows.append({"Variant": r["variant"], "Appliance": a,
                         "Accuracy": m["accuracy"], "Balanced Acc": m["balanced_accuracy"],
                         "Macro-F1": m["macro_f1"], "MAE (W)": m["mae_w"], "RMSE (W)": m["rmse_w"],
                         "ON": m["n_on"], "OFF": m["n_off"], "Grade": m["grade"]})
    return pd.DataFrame(rows)


def make_report(exp_name):
    blob = json.loads((RESULTS_DIR / exp_name / "results.json").read_text())
    meta, results = blob["meta"], blob["results"]
    d = meta["data_cfg"]
    if "regional_homes" in meta:
        return regional_report(meta, results)
    L = []
    L.append(f"# ANN-based Non-Intrusive Load Monitoring (NILM) - Report\n")
    L.append(f"*Experiment `{exp_name}` - generated {meta['created']} - seed {meta['train_cfg']['seed']}*\n")
    L.append("## 1. Problem definition")
    L.append("- **Input (X):** whole-house power only = `grid + solar (+solar2)` in W, as a window of the "
             f"previous **{d['window']} one-minute readings**.")
    L.append("- **Targets (y):** for each appliance, the ON/OFF state and the power (W) at the last minute of the window. "
             "Appliance circuits are used *only* as ground truth, never as input.")
    L.append(f"- **Benchmarks (test set, per appliance):** minimum Acc >= {MIN_T['accuracy']:.0%}, BalAcc >= "
             f"{MIN_T['balanced_accuracy']:.0%}, Macro-F1 >= {MIN_T['macro_f1']}; higher Acc >= {HIGH_T['accuracy']:.0%}, "
             f"BalAcc >= {HIGH_T['balanced_accuracy']:.0%}, Macro-F1 >= {HIGH_T['macro_f1']}.\n")
    L.append("## 2. Data")
    L.append(f"- **Region:** {d['region']}  |  **Period:** {d['start']} to {d['end']}")
    L.append(f"- **Training homes (temporal split):** {', '.join(map(str, d['homes']))}")
    ur = d.get("unseen_region") or d["region"]
    L.append(f"- **Unseen test homes ({ur}):** {', '.join(map(str, d['unseen_homes'])) or 'none'}")
    L.append(f"- **Appliances:** {', '.join(d['appliances'])}\n")
    L.append("**Cleaning:** rows sorted by timestamp, duplicate minutes dropped, re-indexed onto a continuous "
             "1-minute grid, gaps <= 5 min linearly interpolated, longer gaps left missing (windows touching them are "
             "discarded), kW converted to W, negative sensor noise clipped to 0.\n")
    L.append("| Home | raw rows | grid minutes | missing before | missing after |\n|---|---|---|---|---|")
    for h, c in meta["cleaning"].items():
        L.append(f"| {h} | {c['rows_raw']:,} | {c['rows_grid']:,} | {c['missing_before']:,} | {c['missing_after']:,} |")
    L.append("\n**Split (time order, per home):** "
             f"{d['train_frac']:.0%} train / {d['val_frac']:.0%} validation / {1 - d['train_frac'] - d['val_frac']:.0%} test, "
             "plus whole unseen homes. Windows are built *inside* each split so none crosses a boundary. "
             "The input scaler and the ON thresholds were fitted on training data only.\n")
    L.append("**ON thresholds** (2-cluster k-means on log training power, floor 10 W):\n")
    L.append("| Appliance | threshold (W) |\n|---|---|")
    for a, t in meta["thresholds_w"].items():
        L.append(f"| {a} | {t:.1f} |")
    L.append("\n**Class counts (windows):**\n")
    L.append("| Split | " + " | ".join(f"{a} ON / OFF" for a in d["appliances"]) + " |")
    L.append("|---|" + "---|" * len(d["appliances"]))
    for s, cc in meta["class_counts"].items():
        L.append(f"| {s} | " + " | ".join(f"{cc[a]['ON']:,} / {cc[a]['OFF']:,}" for a in d["appliances"]) + " |")
    L.append("\n## 3. Models")
    for r in results:
        desc = DESCRIPTIONS.get(r["model"], {"Rule": "Non-neural step-threshold heuristic.",
                                             "LogReg": "Logistic + ridge regression on window statistics.",
                                             "GBoost": "Histogram gradient boosting on window statistics."}.get(r["model"], ""))
        L.append(f"- **{r['variant']}** ({r['logic']}-task, {r['params']:,} params, "
                 f"{r['train_time_s']:.0f}s train): {desc}")
    tc = meta["train_cfg"]
    L.append(f"\nTraining: Adam lr={tc['lr']}, batch {tc['batch_size']}, <= {tc['epochs']} epochs, early stopping "
             f"(patience {tc['patience']}) on validation loss; loss = class-weighted BCE + Huber on scaled power. "
             "The probability cut-off per appliance is tuned on validation macro-F1 (never on test).\n")
    for split, title in (("test", "4. Results - held-out test (seen homes, future time)"),
                         ("unseen", "5. Generalisation - completely unseen homes")):
        bt = benchmark_table(results, split)
        if bt.empty:
            continue
        L.append(f"## {title}")
        L.append("| Variant | Appliance | Acc | BalAcc | Macro-F1 | MAE W | RMSE W | Grade |\n|---|---|---|---|---|---|---|---|")
        for _, x in bt.iterrows():
            L.append(f"| {x['Variant']} | {x['Appliance']} | {_fmt(x['Accuracy'], 1)} | {_fmt(x['Balanced Acc'], 1)} | "
                     f"{_fmt(x['Macro-F1'])} | {x['MAE (W)']:.1f} | {x['RMSE (W)']:.1f} | {x['Grade']} |")
        L.append("")
    L.append("## 6. Comparison and discussion")
    bt = benchmark_table(results, "test")
    if not bt.empty:
        agg = bt.groupby("Variant")[["Accuracy", "Balanced Acc", "Macro-F1", "MAE (W)"]].mean()
        best = agg["Macro-F1"].idxmax()
        L.append(f"- Best mean test Macro-F1: **{best}** ({agg.loc[best, 'Macro-F1']:.3f}).")
        for v in agg.index:
            n_ex = (bt[bt.Variant == v].Grade == "Excellent").sum()
            n_min = (bt[bt.Variant == v].Grade != "Below").sum()
            L.append(f"- {v}: mean Macro-F1 {agg.loc[v, 'Macro-F1']:.3f}, mean MAE {agg.loc[v, 'MAE (W)']:.1f} W; "
                     f"{n_min}/{len(d['appliances'])} appliances meet minimum, {n_ex} meet the higher targets.")
    L.append("\n## 7. Reproduce")
    L.append("```bash\npip install -r requirements.txt\npython -m nilm.ingest --region austin\n"
             f"python -m nilm.experiment --preset all --name {exp_name}\nstreamlit run app.py\n```")
    return "\n".join(L)


def regional_report(meta, results):
    cfg, train = meta["data_cfg"], meta["train_cfg"]
    name = meta["experiment"]
    lines = [
        "# Multi-region NILM comparison",
        "",
        f"Training regions: {', '.join(meta['train_regions'])}. Seed {cfg['seed']}.",
        "",
        f"Each model receives {cfg['window']} consecutive one-minute whole-home readings "
        "(grid plus available solar, converted to watts). Appliance meters provide ON/OFF and "
        "watts at the last minute. Unmetered appliance labels are excluded from loss and scoring.",
        "",
        "Each participating home's timeline is split 70% / 15% / 15%. In transfer runs, only "
        "training regions contribute to fitting and validation. Approximately 20% "
        "of eligible homes per region are held out completely. Short gaps of up to five minutes "
        "are interpolated within partitions. Windows with missing aggregate values are excluded. "
        "The scaler and physical ON thresholds use training windows only. Probability cutoffs "
        "for learned classifiers use validation macro-F1. The rule baseline uses a fixed decision cutoff.",
        "",
        f"Per-home window settings: training {cfg['train_per_home']}, evaluation {cfg['eval_per_home']} "
        "(0 means all unless a minimum coverage is specified). Windows are sampled evenly over each eligible period. "
        "Power errors are measured at sampled minutes. The Puerto Rico source has sparse appliance "
        "coverage. A regional cell needs metered data and both states to count toward the regional mean.",
        "",
        "| Region | Training homes | Completely unseen homes | Excluded homes |",
        "|---|---|---|---|",
    ]
    for region, h in meta["regional_homes"].items():
        lines.append(f"| {region} | {', '.join(map(str, h['train']))} | "
                     f"{', '.join(map(str, h['unseen']))} | {', '.join(map(str, h['excluded']))} |")
    lines += ["", "Home-level dates, observed target channels and cleaning counts are in `results.json`."]
    coverage = meta.get("coverage_by_region")
    if coverage:
        lines += ["", "### Training coverage by region", "",
                  "| Region | Training homes / source homes | Training windows used | Usable training windows available |",
                  "|---|---:|---:|---:|"]
        for region, counts in coverage.items():
            trained = len(meta["regional_homes"][region]["train"])
            total = counts["raw_homes"]
            lines.append(f"| {region} | {trained}/{total} ({trained / total:.0%}) | "
                         f"{counts['train_windows']:,} | {counts['available_train_windows']:,} |")
        lines += ["", "Home coverage counts participating homes. Window coverage counts sampled minutes "
                  "within the earliest 70% of those homes' timelines."]
    lines += ["", "## Classes and thresholds", "",
              "| Appliance | ON above (W) | Training ON / OFF | Unseen ON / OFF |", "|---|---:|---:|---:|"]
    for appliance, threshold in meta["thresholds_w"].items():
        a = meta["class_counts"]["train"][appliance]
        b = meta["class_counts"]["unseen"][appliance]
        lines.append(f"| {appliance} | {threshold:.1f} | {a['ON']:,} / {a['OFF']:,} | {b['ON']:,} / {b['OFF']:,} |")
    lines += ["", "## Results on completely unseen homes", "",
              "The minimum course benchmarks per appliance are 85% accuracy, 75% balanced accuracy "
              "and 0.70 macro-F1. Higher targets are 90%, 85% and 0.80 respectively.", "",
              "| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | MAE (W) | RMSE (W) | Benchmark |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for result in results:
        for appliance, m in result["splits"]["unseen"].items():
            lines.append(f"| {result['variant']} | {appliance} | {m['accuracy']:.1%} | "
                         f"{m['balanced_accuracy']:.1%} | {m['macro_f1']:.3f} | "
                         f"{m['mae_w']:.1f} | {m['rmse_w']:.1f} | {m['grade']} |")
    lines += ["", "## Performance by region", "",
              "Macro-F1 averaged equally across appliances with both ON and OFF support in each region.", "",
              "| Model | " + " | ".join(meta["regional_homes"]) + " | Training seconds | Inference ms per 1,000 |",
              "|---|" + "---:|" * (len(meta["regional_homes"]) + 2)]
    for result in results:
        regional = result.get("regional_splits", {}).get("unseen", {})
        cells = []
        for region in meta["regional_homes"]:
            values = [m["macro_f1"] for m in regional.get(region, {}).values() if m["n_on"] and m["n_off"]]
            cells.append(f"{sum(values) / len(values):.3f}" if values else "—")
        lines.append(f"| {result['variant']} | " + " | ".join(cells)
                     + f" | {result['train_time_s']:.1f} | {result['infer_ms_per_1k']['unseen']:.1f} |")
    summaries = []
    for result in results:
        unseen = list(result["splits"]["unseen"].values())
        future = list(result["splits"]["test"].values())
        summaries.append({
            "model": result["variant"], "f1": sum(m["macro_f1"] for m in unseen) / len(unseen),
            "future": sum(m["macro_f1"] for m in future) / len(future),
            "mae": sum(m["mae_w"] for m in unseen) / len(unseen),
            "minimum": sum(m["grade"] in {"Minimum", "Excellent"} for m in unseen),
        })
    comparison = [row for row in summaries if row["model"] not in
                  {"CNN1D-noweight", "CNN1D-thr0.5", "CNN1D-focal"}] or summaries
    best = max(comparison, key=lambda row: row["f1"])
    power = min(comparison, key=lambda row: row["mae"])
    lines += ["", "## Model trade-offs", "",
              f"{best['model']} had the highest observed mean unseen-home macro-F1 ({best['f1']:.3f}). "
              f"{power['model']} had the lowest mean unseen-home MAE ({power['mae']:.1f} W). "
              "The table averages appliances equally. Results come from one seeded home split.", "",
              "| Model | Future-time macro-F1 | Unseen macro-F1 | Unseen MAE (W) | Appliances meeting minimum | Trainable parameters |",
              "|---|---:|---:|---:|---:|---:|"]
    params = {r["variant"]: r["params"] for r in results}
    for row in sorted(comparison, key=lambda row: -row["f1"]):
        lines.append(f"| {row['model']} | {row['future']:.3f} | {row['f1']:.3f} | "
                     f"{row['mae']:.1f} | {row['minimum']} | {params[row['model']]:,} |")
    core = {"air1", "furnace1", "drye1", "refrigerator1"}
    if set(cfg["appliances"]) > core:
        lines += ["", "### Original four appliance targets", "",
                  "The overall ranking includes lighting and water heating. This table shows performance "
                  "on the four original appliance targets within the expanded experiment.", "",
                  "| Model | Original-target macro-F1 | Original-target MAE (W) |", "|---|---:|---:|"]
        for result in results:
            if result["variant"] not in {row["model"] for row in comparison}:
                continue
            scores = [m for a, m in result["splits"]["unseen"].items() if a in core]
            lines.append(f"| {result['variant']} | {sum(m['macro_f1'] for m in scores) / len(scores):.3f} | "
                         f"{sum(m['mae_w'] for m in scores) / len(scores):.1f} |")
    transfer_path = RESULTS_DIR / name / "austin_vs_pooled.csv"
    if transfer_path.exists():
        transfer = pd.read_csv(transfer_path)
        mean = transfer[transfer.region == "pooled"].pivot_table(
            index="variant", columns="experiment", values="macro_f1")
        baseline = next(column for column in mean if column != name)
        lines += ["", "## Austin-only versus pooled training", "",
                  "Both fits were scored on the same unseen windows with the pooled training run's "
                  "physical ON thresholds. Each fit kept its own validation-tuned probability cutoffs. "
                  "The change reflects both more homes and wider geographic coverage.", "",
                  "| Model | Austin-only macro-F1 | Pooled macro-F1 | Change |", "|---|---:|---:|---:|"]
        for variant, row in mean.iterrows():
            lines.append(f"| {variant} | {row[baseline]:.3f} | {row[name]:.3f} | "
                         f"{row[name] - row[baseline]:+.3f} |")
    sample_path = RESULTS_DIR / name / "training_size_comparison.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        mean = sample[sample.region == "pooled"].pivot_table(
            index="variant", columns="experiment", values="macro_f1")
        smaller = next(column for column in mean if column != name)
        lines += ["", "## Effect of training sample size", "",
                  "Both runs used the same homes, target circuits, window length and test labels. "
                  "The smaller run used 1,150 training windows per home. The expanded run used 2,500.", "",
                  "| Model | 1,150 per home | 2,500 per home | Change in unseen macro-F1 |", "|---|---:|---:|---:|"]
        for variant, row in mean.iterrows():
            lines.append(f"| {variant} | {row[smaller]:.3f} | {row[name]:.3f} | "
                         f"{row[name] - row[smaller]:+.3f} |")
    techniques = {r["variant"]: r for r in results if r["variant"] in {
        "CNN1D", "CNN1D-noweight", "CNN1D-thr0.5", "CNN1D-focal"}}
    if len(techniques) > 1:
        labels = {"CNN1D": "Weighted BCE, cutoff tuned on validation",
                  "CNN1D-noweight": "Unweighted BCE, cutoff tuned on validation",
                  "CNN1D-thr0.5": "Weighted BCE, fixed 0.5 cutoff",
                  "CNN1D-focal": "Weighted focal loss, cutoff tuned on validation"}
        lines += ["", "## Controlled CNN1D comparison", "",
                  "Architecture, inputs, splits, regression objective, seed and training budget are shared.",
                  "", "| Classification and cutoff | Unseen macro-F1 | Unseen balanced accuracy | Unseen MAE (W) |",
                  "|---|---:|---:|---:|"]
        for variant, result in techniques.items():
            values = pd.DataFrame(result["splits"]["unseen"]).T
            lines.append(f"| {labels[variant]} | {values.macro_f1.mean():.3f} | "
                         f"{values.balanced_accuracy.mean():.3f} | {values.mae_w.mean():.1f} |")
    lines += ["", "The detailed CSV includes validation, future-time and unseen-home splits, "
              "per-region support counts and the four confusion matrix counts. The Streamlit app "
              "shows training curves, confusion matrices and actual versus predicted power.", "",
              "## Reproduce", "", "```powershell",
              f"python -m nilm.experiment --preset multiregion --name {name} --epochs {train['epochs']} "
              f"--max-train {train['max_train']} --train-per-home {cfg['train_per_home']} "
              f"--eval-per-home {cfg['eval_per_home']} --regions {' '.join(cfg['regions'])} "
              f"--window {cfg['window']} --appliances {' '.join(cfg['appliances'])} "
              f"--only {','.join(r['variant'] for r in results)}"
              + (" --techniques" if len(techniques) > 1 else "")
              + (f" --minimum-window-coverage {cfg['minimum_window_coverage']}"
                 if cfg.get("minimum_window_coverage") else "")
              + (f" --train-regions {' '.join(meta['train_regions'])}"
                 if set(meta["train_regions"]) != set(cfg["regions"]) else ""),
              f"python -m nilm.traces {name}", f"python -m nilm.dashboard {name}", "streamlit run app.py", "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    txt = make_report(name)
    (RESULTS_DIR / name / "REPORT.md").write_text(txt, encoding="utf-8")
    print(f"Saved results/{name}/REPORT.md")
