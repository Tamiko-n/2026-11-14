"""Export a self-contained interactive comparison from the saved test results."""
import argparse

import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .config import RESULTS_DIR


def make_dashboard(name):
    directory = RESULTS_DIR / name
    regional = pd.read_csv(directory / "regional_metrics.csv")
    unseen = regional[regional.split.eq("unseen") & regional.n_on.gt(0) & regional.n_off.gt(0)]
    pooled = pd.read_csv(directory / "metrics.csv")
    pooled = pooled[pooled.split.eq("unseen")]
    summary = pooled.groupby("variant", as_index=False)[["macro_f1", "mae_w", "train_time_s", "params"]].mean()
    figures = [
        px.scatter(summary, x="mae_w", y="macro_f1", text="variant", color="train_time_s", hover_data=["params"],
                   labels={"mae_w": "Mean MAE (W)", "macro_f1": "Mean macro-F1",
                           "train_time_s": "Training seconds"}, title="Unseen-home classification and power error"),
        px.bar(pooled, x="variant", y="macro_f1", facet_col="appliance", facet_col_wrap=3, height=650,
               labels={"variant": "Model", "macro_f1": "Macro-F1"},
               hover_data=["balanced_accuracy", "mae_w", "rmse_w", "grade"],
               title="Unseen-home classification by appliance"),
        px.imshow(unseen.groupby(["variant", "region"]).macro_f1.mean().unstack(), text_auto=".3f",
                  labels={"x": "Region", "y": "Model", "color": "Macro-F1"},
                  title="Unseen-home macro-F1 by region"),
    ]
    figures[0].update_traces(textposition="top center")
    techniques = pooled[pooled.variant.isin(["CNN1D", "CNN1D-noweight", "CNN1D-thr0.5", "CNN1D-focal"])]
    if techniques.variant.nunique() > 1:
        figures.append(px.bar(
            techniques, x="variant", y="macro_f1", facet_col="appliance", facet_col_wrap=3,
            hover_data=["balanced_accuracy", "mae_w", "rmse_w"],
            labels={"variant": "Training technique", "macro_f1": "Macro-F1"},
            title="CNN1D classification loss and probability cutoff", height=650,
        ))
    history = pd.read_csv(directory / "training_log.csv")
    if not history.empty:
        figures.append(px.line(history, x="epoch", y="val_macro_f1", color="variant", markers=True,
                               labels={"val_macro_f1": "Validation macro-F1 at cutoff 0.5"},
                               title="Neural training curves"))
    transfer_path = directory / "austin_vs_pooled.csv"
    if transfer_path.exists():
        transfer = pd.read_csv(transfer_path)
        transfer = transfer[(transfer.region != "pooled") & transfer.n_on.gt(0) & transfer.n_off.gt(0)]
        transfer = transfer.groupby(["region", "experiment", "variant"], as_index=False).macro_f1.mean()
        figures.append(px.bar(transfer, x="variant", y="macro_f1", color="experiment", barmode="group",
                              facet_col="region", labels={"experiment": "Training data"},
                              title="Austin-only versus pooled training on matching test labels"))
    size_path = directory / "training_size_comparison.csv"
    if size_path.exists():
        sizes = pd.read_csv(size_path)
        sizes = sizes[sizes.region.eq("pooled")]
        sizes["windows_per_home"] = sizes.experiment.map({name: "2,500", "expanded_small": "1,150"})
        figures.append(px.bar(sizes, x="variant", y="macro_f1", color="windows_per_home",
                              facet_col="appliance", facet_col_wrap=3, barmode="group",
                              title="Training sample size on matching test labels",
                              labels={"windows_per_home": "Training windows per home"}, height=650))
    traces = pd.read_csv(directory / "prediction_traces.csv", parse_dates=["time"])
    shown = traces[traces.appliance.eq("refrigerator1") & traces.actual_w.notna() &
                   traces.variant.isin(["MLP", "CNN1D", "GBoost"])].copy()
    shown["first_actual"] = ~shown.duplicated(["region", "time"])
    actual = shown[shown.first_actual][["region", "time", "actual_w"]].rename(columns={"actual_w": "watts"})
    actual["variant"] = "Actual"
    predicted = shown[["region", "time", "variant", "predicted_w"]].rename(columns={"predicted_w": "watts"})
    figures.append(px.line(pd.concat([actual, predicted]), x="time", y="watts", color="variant",
                           facet_row="region", height=800, title="Refrigerator power on unseen homes",
                           labels={"time": "Time (UTC)", "watts": "Watts", "variant": "Series"}))
    figures[-1].update_xaxes(matches=None, showticklabels=True)
    charts = "\n".join(pio.to_html(fig, full_html=False, include_plotlyjs=False) for fig in figures)
    output = directory / "comparison.html"
    output.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Multi-region NILM model comparison</title>'
        '<style>body{font-family:system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem}'
        'h1{font-size:2rem}p{max-width:75ch;color:#38445a}</style>'
        f'<script>{get_plotlyjs()}</script></head><body><h1>Multi-region NILM model comparison</h1>'
        '<p>These results use homes held out completely in each region. Regional means include only '
        'metered appliances with both ON and OFF examples. Open the detailed CSV for class counts '
        'and confusion matrices.</p>'
        f'{charts}</body></html>', encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="multiregion")
    print(make_dashboard(parser.parse_args().name))
