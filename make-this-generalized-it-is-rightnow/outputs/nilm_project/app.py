"""NILM Studio - Streamlit UI for the ANN NILM CCP.

Run:  streamlit run app.py
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nilm.config import RESULTS_DIR, REGIONS, region_cache
from nilm.regional import RegionalConfig, build_regional, read_home, interpolate_short_gaps
from nilm.data import DataConfig, load_home
from nilm.engine import TrainConfig, MIN_T, HIGH_T
from nilm.experiment import run_experiment, list_experiments, load_results, PRESETS, REGIONAL_TECHNIQUES
from nilm.models import MODELS, DESCRIPTIONS
from nilm.report import make_report, benchmark_table

st.set_page_config(page_title="NILM Studio", page_icon=None, layout="wide")

GRADE_COLOR = {"Excellent": "#1b9e5a", "Minimum": "#d4a017", "Below": "#c0392b"}
PALETTE = px.colors.qualitative.Set2

st.markdown("""
<style>
.block-container{padding-top:1.4rem}
.kpi{border:1px solid rgba(37,99,235,.18);border-radius:14px;padding:.8rem 1rem;
 background:linear-gradient(135deg,rgba(37,99,235,.08),rgba(16,185,129,.06));box-shadow:0 1px 3px rgba(0,0,0,.05)}
.brand{font-size:1.7rem;font-weight:800;letter-spacing:-.5px;color:#1e3a8a}.brand span{color:#10b981}
h1,h2{letter-spacing:-.5px}
h2{border-left:5px solid #2563eb;padding-left:.6rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b)}
section[data-testid="stSidebar"] *{color:#e2e8f0}
section[data-testid="stSidebar"] .brand{color:#93c5fd}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:.45rem .7rem;border-radius:8px;margin:2px 0;width:100%}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(147,197,253,.12)}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:#2563eb}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#0f172a}
.kpi .v{font-size:1.6rem;font-weight:700}
.kpi .l{font-size:.8rem;opacity:.7}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def coverage(region):
    if not region:
        return None
    p = region_cache(region) / "_coverage.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def cached_home(region, hid, apps, start, end):
    df, rep = load_home(region, hid, list(apps), start, end)
    return df, rep


def results_frame(blob):
    rows = []
    for r in blob["results"]:
        for split, d in r["splits"].items():
            for a, m in d.items():
                rows.append({"variant": r["variant"], "model": r["model"], "logic": r["logic"], "split": split,
                             "appliance": a, **{k: v for k, v in m.items() if k != "cm"},
                             "params": r["params"], "train_time_s": r["train_time_s"],
                             "infer_us": r.get("infer_ms_per_1k", {}).get(split, np.nan)})
    return pd.DataFrame(rows)


def kpi(col, label, value):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div></div>",
                 unsafe_allow_html=True)


def style_grade(df):
    def c(v):
        return f"background-color:{GRADE_COLOR.get(v, '')};color:white;font-weight:600" if v in GRADE_COLOR else ""
    fmt = {k: "{:.1%}" for k in ("Accuracy", "Balanced Acc") if k in df}
    fmt.update({k: "{:.3f}" for k in ("Macro-F1",) if k in df})
    fmt.update({k: "{:.1f}" for k in ("MAE (W)", "RMSE (W)") if k in df})
    fmt.update({k: "{:,}" for k in ("ON", "OFF") if k in df})
    return df.style.format(fmt).map(c, subset=["Grade"]) if "Grade" in df else df.style.format(fmt)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("<div class='brand'>NILM <span>Studio</span></div>", unsafe_allow_html=True)
st.sidebar.caption("ANN Non-Intrusive Load Monitoring · CT-466 CCP")
exps = list_experiments()
PAGES = ['Overview', 'Data Explorer', 'Train', 'Results', 'Comparison', 'Generalisation', 'Live Disaggregation', 'Report']
page = st.sidebar.radio("Navigation", PAGES, key="nav")

default_exp = next((name for name in ("expanded", "multiregion") if name in exps), None)
sel_exp = st.sidebar.selectbox("Experiment to inspect", exps, index=exps.index(default_exp) if default_exp else 0) \
    if exps else None
st.sidebar.markdown("---")
st.sidebar.markdown("**Benchmarks (per appliance, test)**")
st.sidebar.markdown(f"- Minimum: Acc ≥ {MIN_T['accuracy']:.0%}, BalAcc ≥ {MIN_T['balanced_accuracy']:.0%}, "
                    f"F1 ≥ {MIN_T['macro_f1']}\n- Higher: Acc ≥ {HIGH_T['accuracy']:.0%}, "
                    f"BalAcc ≥ {HIGH_T['balanced_accuracy']:.0%}, F1 ≥ {HIGH_T['macro_f1']}")

# ================================================================== OVERVIEW
if page == "Overview":
    st.header("ANN-based Non-Intrusive Load Monitoring")
    if True:
        st.markdown("""
A smart meter sees only the **total** power of a home. NILM recovers **which appliances are ON** and
**how much power each draws**, from that single signal.

| | |
|---|---|
| **Input X** | whole-house power (`grid + solar`) — a window of the last *W* one-minute readings |
| **Targets y** | per appliance: ON/OFF label and power (W) at the last minute |
| **Ground truth** | Pecan Street appliance circuits (never used as input) |
| **Key constraints** | time-ordered split · thresholds & scaler fitted on train only · windows never cross splits |
""")
    if True:
        st.markdown("**Pipeline**")
        st.graphviz_chart("""
digraph{rankdir=LR;node[shape=box,style="rounded,filled",fillcolor="#eef3fb",fontname=Helvetica,fontsize=10]
a[label="Raw regional CSVs\\n(Austin/CA/NY/PR)"];b[label="Ingest → per-home parquet\\n+ coverage survey"];
c[label="Clean: sort, dedupe,\\n1-min grid, gap fill ≤5 min"];d[label="Time split 70/15/15\\n+ unseen homes"];
e[label="Train-only: ON thresholds\\n(k-means) + scaler"];f[label="Sliding windows\\n(within split)"];
g[label="MLP · CNN1D · LSTM · GRU · CNN-GRU\\n+ Rule · LogReg · GBoost"];h[label="Val: early stop +\\ndecision threshold"];
i[label="Test & unseen homes:\\nAcc · BalAcc · F1 · MAE · RMSE"];a->b->c->d->e->f->g->h->i}""")
    st.subheader("What is compared")
    st.markdown("""
- **Architectures:** MLP (baseline) vs 1D-CNN, LSTM, GRU, CNN-GRU hybrid.
- **Different logic:** neural vs non-neural (step-threshold rule, logistic regression, gradient boosting);
  multi-task (one net, all appliances) vs single-task (one net per appliance); class-weighted vs unweighted loss;
  validation-tuned vs fixed 0.5 decision threshold; window length 15/30/60/120 min.
- **Generalisation:** future time on seen homes → completely unseen homes → homes from another region.
""")
    st.subheader("Model zoo")
    st.dataframe(pd.DataFrame({"Model": list(DESCRIPTIONS), "Idea": list(DESCRIPTIONS.values())}),
                 hide_index=True, use_container_width=True)

# ================================================================== DATA EXPLORER
if page == "Data Explorer":
    st.header("Data Explorer")
    regions_ready = [r for r in REGIONS if (region_cache(r) / "_coverage.csv").exists()]
    if not regions_ready:
        st.warning("No region ingested yet. Run `python -m nilm.ingest --region all`.")
    else:
        region = st.selectbox("Region", regions_ready)
        cov = coverage(region)
        on_cols = [c for c in cov.columns if c.startswith("on%_")]
        st.markdown(f"**{len(cov)} homes**. Cells = % of minutes the circuit is above 30 W "
                    "(blank = circuit not metered). Good NILM targets have both ON and OFF time.")
        common = cov[on_cols].notna().sum().sort_values(ascending=False)
        keep = [c for c in common.index if common[c] >= 3][:18]
        hm = cov.set_index("dataid")[keep].rename(columns=lambda c: c[4:])
        hm.index = "home " + hm.index.astype(str)
        fig = px.imshow(hm, aspect="auto", color_continuous_scale="Viridis", text_auto=".0f",
                        labels={"color": "% ON"})
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        hid = c1.selectbox("Home", cov["dataid"].tolist())
        apps_avail = [c[4:] for c in on_cols if not np.isnan(cov.set_index("dataid").loc[hid, c])]
        apps = c2.multiselect("Appliances", apps_avail,
                              default=[a for a in ("air1", "furnace1", "refrigerator1") if a in apps_avail][:3])
        row = cov.set_index("dataid").loc[hid]
        d0 = pd.Timestamp(row["start"]).date()
        d1 = pd.Timestamp(row["end"]).date()
        day = c3.date_input("Day", value=min(d1, d0 + pd.Timedelta(days=120)),
                            min_value=d0, max_value=d1, key=f"explore_{region}_{hid}")
        if apps:
            df, rep = cached_home(region, hid, tuple(apps), str(day), str(day + pd.Timedelta(days=1)))
            fig = go.Figure()
            fig.add_scatter(x=df.index, y=df["aggregate"], name="whole-house (input X)",
                            line=dict(color="#555", width=1.5))
            for i, a in enumerate(apps):
                fig.add_scatter(x=df.index, y=df[a], name=f"{a} (target y)", line=dict(width=1.2, color=PALETTE[i]))
            fig.update_layout(height=420, yaxis_title="Power (W)", legend=dict(orientation="h"),
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            m = st.columns(4)
            kpi(m[0], "minutes", f"{rep['rows_grid']:,}")
            kpi(m[1], "missing (raw)", f"{rep['missing_before']:,}")
            kpi(m[2], "missing after fill", f"{rep['missing_after']:,}")
            kpi(m[3], "mean whole-house", f"{df['aggregate'].mean():.0f} W")
            st.markdown("**Power distributions (log)** — the valley between the two modes is where the "
                        "k-means ON threshold lands.")
            long = df[apps].melt(var_name="appliance", value_name="W").dropna()
            long = long[long.W > 1]
            long["log10 W"] = np.log10(long["W"])
            fig = px.histogram(long, x="log10 W", color="appliance", nbins=80, barmode="overlay", opacity=.6)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ================================================================== TRAIN
train_mode = st.radio("Training scope", ["Pooled regions", "Single region"], horizontal=True) if page == "Train" else None
if train_mode == "Pooled regions":
    st.header("Train on multiple regions")
    ready = [r for r in REGIONS if (region_cache(r) / "_coverage.csv").exists()]
    if len(ready) < len(REGIONS):
        st.info("Prepare the regional caches using the commands in README.md.")
    regions = st.multiselect("Regions", ready, default=ready)
    techniques = st.checkbox("Compare classification techniques", value=True)
    variants = PRESETS["architectures"] + PRESETS["classical"]
    if techniques:
        variants += [v for v in PRESETS["logic"] if v["name"] in REGIONAL_TECHNIQUES]
    models = st.multiselect("Models", [v["name"] for v in variants], default=[v["name"] for v in variants])
    epochs = st.number_input("Maximum epochs", 1, 100, 5)
    window = st.selectbox("Input window (minutes)", [15, 30, 60, 90, 120], index=1)
    windows_per_home = st.number_input("Training windows per home (0 uses all)", min_value=0, value=2500, step=500)
    apps = st.multiselect("Appliances and circuits",
                          ["air1", "furnace1", "drye1", "refrigerator1", "lights_plugs1", "waterheater1"],
                          default=["air1", "furnace1", "drye1", "refrigerator1", "lights_plugs1", "waterheater1"])
    name = st.text_input("Experiment name", f"pooled_{time.strftime('%m%d_%H%M')}")
    if st.button("Train pooled models", disabled=not regions or not models or not apps, type="primary"):
        import torch
        torch.set_num_threads(4)
        logbox, bar = st.empty(), st.progress(0.0)
        lines = []

        def regional_log(message):
            lines.append(message)
            logbox.code("\n".join(lines[-20:]))

        try:
            cfg = RegionalConfig(regions=regions, appliances=apps, window=window,
                                 train_per_home=int(windows_per_home), eval_per_home=600)
            data = build_regional(cfg, log=regional_log)
            run_experiment(name, cfg, TrainConfig(epochs=int(epochs), max_train=0),
                           [v for v in variants if v["name"] in models],
                           log=regional_log, progress=bar.progress, data=data)
            (RESULTS_DIR / name / "REPORT.md").write_text(make_report(name), encoding="utf-8")
            st.success("Training finished. Refresh and select the experiment in the sidebar.")
        except Exception as e:
            st.exception(e)

if train_mode == "Single region":
    st.header("Configure & train an experiment")
    regions_ready = [r for r in REGIONS if (region_cache(r) / "_coverage.csv").exists()]
    if not regions_ready:
        st.warning("Ingest a region first.")
    else:
        dflt = DataConfig()
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Training region", regions_ready, key="tr_region")
        cov = coverage(region)
        homes_all = cov["dataid"].tolist()
        apps_all = sorted(c[4:] for c in cov.columns if c.startswith("on%_"))
        homes = c1.multiselect("Training homes (≥5)", homes_all,
                               default=[h for h in dflt.homes if h in homes_all] or homes_all[:5])
        apps = c2.multiselect("Appliances (≥3)", apps_all, default=[a for a in dflt.appliances if a in apps_all])
        ok_homes = [h for h in homes_all if h not in homes and
                    all(not np.isnan(cov.set_index("dataid").loc[h].get(f"on%_{a}", np.nan)) for a in apps)]
        un_region = c3.selectbox("Unseen-home region", regions_ready, index=regions_ready.index(region))
        if un_region == region:
            unseen = c3.multiselect("Unseen test homes", ok_homes,
                                    default=[h for h in dflt.unseen_homes if h in ok_homes])
        else:
            ucov = coverage(un_region)
            ucand = [h for h in ucov["dataid"] if all(
                not np.isnan(ucov.set_index("dataid").loc[h].get(f"on%_{a}", np.nan)) for a in apps)]
            unseen = c3.multiselect("Unseen test homes", ucand, default=ucand[:1])
        c2.caption("Homes eligible = all chosen appliances metered.")
        d1, d2, d3, d4 = st.columns(4)
        start = d1.text_input("Start date", dflt.start)
        end = d2.text_input("End date", dflt.end)
        if un_region != region:
            ustart = d1.text_input("Unseen start", "")
            uend = d2.text_input("Unseen end", "")
        else:
            ustart = uend = ""
        window = d3.select_slider("Window (minutes)", [15, 30, 60, 90, 120], value=60)
        stride = d4.number_input("Train stride", 1, 10, 2)
        st.markdown("**Variants**")
        v1, v2, v3 = st.columns(3)
        nets = v1.multiselect("Neural nets (multi-task)", list(MODELS), default=["MLP", "CNN1D", "GRU"])
        classical = v2.multiselect("Non-neural logic", ["Rule", "LogReg", "GBoost"], default=["GBoost"])
        logic = v3.multiselect("Logic ablations", [v["name"] for v in PRESETS["logic"]], default=[])
        t1, t2, t3, t4 = st.columns(4)
        epochs = t1.number_input("Max epochs", 1, 100, 12)
        bs = t2.selectbox("Batch size", [256, 512, 1024], index=1)
        lr = t3.select_slider("Learning rate", [3e-4, 1e-3, 3e-3], value=1e-3)
        seed = t4.number_input("Seed", 0, 9999, 42)
        name = st.text_input("Experiment name", f"exp_{time.strftime('%m%d_%H%M')}")
        variants = [{"name": n, "model": n} for n in nets] + [{"name": n, "model": n} for n in classical] + \
                   [v for v in PRESETS["logic"] if v["name"] in logic]
        problems = []
        if len(homes) < 5:
            problems.append("choose at least 5 training homes")
        if len(apps) < 3:
            problems.append("choose at least 3 appliances")
        if not variants:
            problems.append("choose at least one variant")
        if problems:
            st.info("To run: " + "; ".join(problems))
        if st.button("Run experiment", type="primary", disabled=bool(problems)):
            dcfg = DataConfig(region=region, homes=homes, unseen_homes=unseen, appliances=apps, start=start,
                              end=end, window=int(window), train_stride=int(stride),
                              unseen_region="" if un_region == region else un_region,
                              unseen_start=ustart, unseen_end=uend)
            tcfg = TrainConfig(epochs=int(epochs), batch_size=int(bs), lr=float(lr), seed=int(seed))
            logbox = st.empty()
            bar = st.progress(0.0)
            lines = []

            def log(msg):
                lines.append(str(msg))
                logbox.code("\n".join(lines[-25:]))

            try:
                run_experiment(name, dcfg, tcfg, variants, log=log, progress=lambda f: bar.progress(min(f, 1.0)))
                (RESULTS_DIR / name / "REPORT.md").write_text(make_report(name), encoding="utf-8")
                st.success(f"Done → results/{name}. Pick it in the sidebar.")
                st.cache_data.clear()
            except Exception as e:
                st.exception(e)

# ================================================================== RESULTS
blob = load_results(sel_exp) if sel_exp else None
if page == "Results":
    st.header("Results")
    if not blob:
        st.info("No experiment yet. Train one on the Train page or run `python -m nilm.experiment`.")
    else:
        meta, res = blob["meta"], blob["results"]
        dc = meta["data_cfg"]
        k = st.columns(5)
        homes = meta.get("regional_homes")
        kpi(k[0], "training regions", len(meta["train_regions"]) if homes else dc["region"])
        kpi(k[1], "training homes", sum(len(h["train"]) for h in homes.values()) if homes else len(dc["homes"]))
        kpi(k[2], "unseen homes", sum(len(h["unseen"]) for h in homes.values()) if homes else len(dc["unseen_homes"]))
        kpi(k[3], "appliances", len(dc["appliances"]))
        kpi(k[4], "window", f"{dc['window']} min")
        split = st.radio("Evaluation set", [s for s in ("unseen", "test", "val") if s in res[0]["splits"]],
                         horizontal=True, format_func={"test": "Test (seen homes, future)",
                                                       "unseen": "Unseen homes", "val": "Validation"}.get)
        bt = benchmark_table(res, split)
        st.subheader("Benchmark table")
        st.dataframe(style_grade(bt), hide_index=True, use_container_width=True)
        variant = st.selectbox("Variant", [r["variant"] for r in res])
        r = next(x for x in res if x["variant"] == variant)
        st.subheader(f"{variant}: per-appliance detail")
        cols = st.columns(len(dc["appliances"]))
        for c, a in zip(cols, dc["appliances"]):
            m = r["splits"][split].get(a)
            if m is None:
                c.caption(f"{a} has no measured targets in this split")
                continue
            cm = np.array(m["cm"])
            fig = px.imshow(cm, text_auto=True, x=["pred OFF", "pred ON"], y=["true OFF", "true ON"],
                            color_continuous_scale="Blues")
            fig.update_layout(height=260, coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0),
                              title=f"{a} · {m['grade']}")
            c.plotly_chart(fig, use_container_width=True)
            c.caption(f"Acc {m['accuracy']:.1%} · BalAcc {m['balanced_accuracy']:.1%} · F1 {m['macro_f1']:.3f}  \n"
                      f"MAE {m['mae_w']:.1f} W · RMSE {m['rmse_w']:.1f} W · cut-off p>{r['decision_thresholds'][a]:.2f}")
        st.subheader("Actual vs predicted appliance power")
        pz = RESULTS_DIR / sel_exp / f"{variant}_preds.npz"
        hours = st.slider("Hours shown", 6, 72, 24) if not homes else None
        regional_trace = RESULTS_DIR / sel_exp / "prediction_traces.csv"
        if homes and regional_trace.exists():
            trace = pd.read_csv(regional_trace, parse_dates=["time"])
            region = st.selectbox("Trace region", trace.region.unique())
            trace = trace[(trace.region == region) & (trace.variant == variant)]
            if not trace.empty:
                st.caption("Example from a completely unseen home. The trace illustrates model behavior at one-minute resolution.")
                fig = px.line(trace, x="time", y=["actual_w", "predicted_w"], facet_row="appliance",
                              height=210 * trace.appliance.nunique(), labels={"value": "Watts", "variable": "Series"})
                st.plotly_chart(fig, use_container_width=True)
        elif pz.exists() and not homes:
            P = np.load(pz)
            from nilm.data import build_dataset  # noqa
            gt_path = RESULTS_DIR / sel_exp / "_truth.npz"
            truth = np.load(gt_path) if gt_path.exists() else None
            for i, a in enumerate(dc["appliances"]):
                key = f"{split}|{a}|power"
                if key not in P:
                    continue
                n = hours * 60 if hours else len(P[key])
                fig = go.Figure()
                x = truth[f"{split}|time"][:n] if truth is not None and f"{split}|time" in truth else np.arange(n)
                if truth is not None and f"{split}|{a}" in truth:
                    fig.add_scatter(x=x, y=truth[f"{split}|{a}"][:n], name="actual", line=dict(color="#333", width=1.4))
                fig.add_scatter(x=x, y=P[key][:n], name="predicted", line=dict(color=PALETTE[i], width=1.4))
                fig.update_layout(height=240, title=a, yaxis_title="W", xaxis_title="Time (UTC)" if homes else "minute",
                                  margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Training curves")
        hist = pd.DataFrame(r["history"])
        if hist.empty:
            st.caption("Non-neural variant - no epochs.")
        else:
            c1, c2 = st.columns(2)
            f1 = px.line(hist.melt(id_vars=["epoch", "appliances"], value_vars=["train_loss", "val_loss"]),
                         x="epoch", y="value", color="variable", line_dash="appliances", markers=True)
            f1.update_layout(height=300, title="Loss", margin=dict(l=0, r=0, t=30, b=0))
            c1.plotly_chart(f1, use_container_width=True)
            f2 = px.line(hist, x="epoch", y="val_macro_f1", color="appliances", markers=True)
            f2.update_layout(height=300, title="Validation macro-F1 (p>0.5)", margin=dict(l=0, r=0, t=30, b=0))
            c2.plotly_chart(f2, use_container_width=True)

# ================================================================== COMPARISON
if page == "Comparison":
    st.header("Model & logic comparison")
    chosen = st.multiselect("Experiments to pool", exps, default=[sel_exp] if sel_exp else [])
    frames = []
    for e in chosen:
        b = load_results(e)
        if b:
            f = results_frame(b)
            f["experiment"] = e
            f["window"] = b["meta"]["data_cfg"]["window"]
            frames.append(f)
    if frames:
        df = pd.concat(frames)
        df["label"] = df["variant"] + np.where(df["experiment"].nunique() > 1, " @" + df["experiment"], "")
        split = st.radio("Set", sorted(df.split.unique(), key=["test", "unseen", "val"].index),
                         horizontal=True, key="cmp_split")
        d = df[df.split == split]
        metric = st.selectbox("Metric", ["macro_f1", "balanced_accuracy", "accuracy", "mae_w", "rmse_w", "sae",
                                         "precision_on", "recall_on"])
        fig = px.bar(d, x="appliance", y=metric, color="label", barmode="group", text_auto=".3f" if
                     metric not in ("mae_w", "rmse_w") else ".0f", color_discrete_sequence=px.colors.qualitative.Bold)
        if metric in MIN_T:
            fig.add_hline(MIN_T[metric], line_dash="dot", annotation_text="minimum")
            fig.add_hline(HIGH_T[metric], line_dash="dash", annotation_text="higher")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Leaderboard (mean over appliances)")
        lb = d.groupby("label").agg(accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
                                    macro_f1=("macro_f1", "mean"), mae_w=("mae_w", "mean"), rmse_w=("rmse_w", "mean"),
                                    min_pass=("grade", lambda g: int((g != "Below").sum())),
                                    high_pass=("grade", lambda g: int((g == "Excellent").sum())),
                                    params=("params", "first"), train_time_s=("train_time_s", "first"),
                                    infer_us=("infer_us", "first")).sort_values("macro_f1", ascending=False)
        st.dataframe(lb.style.format({"accuracy": "{:.1%}", "balanced_accuracy": "{:.1%}", "macro_f1": "{:.3f}",
                                      "mae_w": "{:.1f}", "rmse_w": "{:.1f}", "params": "{:,.0f}",
                                      "train_time_s": "{:.0f}", "infer_us": "{:.1f}"})
                     .background_gradient(subset=["macro_f1"], cmap="Greens")
                     .background_gradient(subset=["mae_w"], cmap="Reds_r"), use_container_width=True)
        st.caption("infer_us = inference time per window in microseconds (CPU). min_pass/high_pass = appliances "
                   "meeting the minimum / higher CCP targets.")
        c1, c2 = st.columns(2)
        radar = lb.reset_index()
        figr = go.Figure()
        axes = ["accuracy", "balanced_accuracy", "macro_f1", "precision", "recall"]
        pr = d.groupby("label")[["precision_on", "recall_on"]].mean()
        for i, (_, x) in enumerate(radar.iterrows()):
            vals = [x.accuracy, x.balanced_accuracy, x.macro_f1, pr.loc[x.label, "precision_on"],
                    pr.loc[x.label, "recall_on"]]
            figr.add_scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=x.label, fill="toself", opacity=.45)
        figr.update_layout(height=420, title="Classification profile", polar=dict(radialaxis=dict(range=[0.4, 1])),
                           margin=dict(l=30, r=30, t=40, b=0))
        c1.plotly_chart(figr, use_container_width=True)
        fige = px.scatter(radar, x="train_time_s", y="macro_f1", size=np.sqrt(radar.params.fillna(0) + 1000),
                          color="label", text="label", log_x=True)
        fige.update_traces(textposition="top center")
        fige.update_layout(height=420, title="Accuracy vs training cost (bubble = √params)", showlegend=False,
                           margin=dict(l=0, r=0, t=40, b=0))
        c2.plotly_chart(fige, use_container_width=True)

        if df["window"].nunique() > 1:
            st.subheader("Window-length ablation")
            w = df[df.split == split].groupby(["window", "model"])[metric].mean().reset_index()
            st.plotly_chart(px.line(w, x="window", y=metric, color="model", markers=True).update_layout(height=320),
                            use_container_width=True)
        st.subheader("Full metric table")
        st.dataframe(d.drop(columns=["label"]), hide_index=True, use_container_width=True)
        st.download_button("Download CSV", d.to_csv(index=False), "comparison.csv")

# ================================================================== GENERALISATION
if page == "Generalisation":
    st.header("Generalisation: does it work beyond the training data?")
    st.markdown("Each bar shows the same model on progressively harder data: **val** (seen homes, tuning period) → "
                "**test** (seen homes, *future* weeks) → **unseen** (homes never seen; possibly another region).")
    frames = []
    for e in exps:
        b = load_results(e)
        if b:
            f = results_frame(b)
            f["experiment"] = e
            dc = b["meta"]["data_cfg"]
            f["unseen_from"] = dc.get("unseen_region") or dc.get("region") or "Pooled regions"
            frames.append(f)
    if frames:
        g = pd.concat(frames)
        e = st.selectbox("Experiment", exps, index=exps.index(sel_exp) if sel_exp in exps else 0, key="gen_exp")
        g1 = g[g.experiment == e]
        metric = st.selectbox("Metric", ["macro_f1", "balanced_accuracy", "accuracy", "mae_w"], key="gen_m")
        fig = px.bar(g1, x="variant", y=metric, color="split", facet_col="appliance", barmode="group",
                     category_orders={"split": ["val", "test", "unseen"]},
                     color_discrete_map={"val": "#9ecae1", "test": "#3182bd", "unseen": "#e6550d"})
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        piv = g1.pivot_table(index="variant", columns="split", values=metric, aggfunc="mean")
        if {"test", "unseen"} <= set(piv.columns):
            piv["drop test→unseen"] = piv["test"] - piv["unseen"]
            st.dataframe(piv.style.format("{:.3f}").background_gradient(subset=["drop test→unseen"], cmap="Reds"),
                         use_container_width=True)
        path = RESULTS_DIR / e / "regional_metrics.csv"
        if path.exists():
            st.subheader("Performance by region")
            regional = pd.read_csv(path)
            split = st.selectbox("Evaluation split", ["unseen", "test", "val"])
            appliance = st.selectbox("Appliance", sorted(regional.appliance.unique()))
            d = regional[(regional.split == split) & (regional.appliance == appliance)]
            heatmap = d.pivot(index="variant", columns="region", values=metric)
            st.plotly_chart(px.imshow(heatmap, text_auto=".3f", aspect="auto"), use_container_width=True)
            st.dataframe(d, hide_index=True, use_container_width=True)
            st.download_button("Download regional metrics", path.read_bytes(), "regional_metrics.csv")
        transfer_path = RESULTS_DIR / e / "austin_vs_pooled.csv"
        if transfer_path.exists():
            st.subheader("Austin-only versus pooled training")
            transfer = pd.read_csv(transfer_path)
            matching = transfer[(transfer.region != "pooled") & (transfer.n_on > 0) & (transfer.n_off > 0)]
            averages = matching.groupby(["region", "variant", "experiment"], as_index=False)[metric].mean()
            st.plotly_chart(px.bar(averages, x="variant", y=metric, color="experiment", facet_col="region",
                                   barmode="group", labels={"experiment": "Training data"}),
                            use_container_width=True)
            st.download_button("Download matching test scores", transfer_path.read_bytes(), "austin_vs_pooled.csv")
        size_path = RESULTS_DIR / e / "training_size_comparison.csv"
        if size_path.exists():
            st.subheader("Training sample size")
            size = pd.read_csv(size_path)
            d = size[size.region == "pooled"]
            st.plotly_chart(px.bar(d, x="variant", y=metric, color="experiment", facet_col="appliance",
                                   facet_col_wrap=3, barmode="group"), use_container_width=True)
            st.download_button("Download sample-size scores", size_path.read_bytes(), size_path.name)
        cross = g[(g.split == "unseen")].groupby(["experiment", "unseen_from", "variant"])[metric].mean().reset_index()
        if cross.unseen_from.nunique() > 1:
            st.subheader("Cross-region transfer")
            st.plotly_chart(px.bar(cross, x="variant", y=metric, color="unseen_from", barmode="group",
                                   hover_data=["experiment"]).update_layout(height=360), use_container_width=True)

# ================================================================== LIVE
if page == "Live Disaggregation":
    st.header("Live disaggregation on any home / day")
    st.caption("Loads a trained model, feeds it ONLY the whole-house signal of the chosen day, and overlays the "
               "predicted appliance power on the metered truth.")
    if not blob:
        st.info("Train an experiment first.")
    else:
        import torch
        from nilm import models as M
        from nilm.classical import predict_classical  # noqa
        meta, res = blob["meta"], blob["results"]
        dc = meta["data_cfg"]
        nn_variants = [r for r in res if r["model"] in MODELS and r["logic"] == "multi"]
        if not nn_variants:
            st.info("Need a multi-task neural variant in this experiment.")
        else:
            c1, c2, c3 = st.columns(3)
            vname = c1.selectbox("Model", [r["variant"] for r in nn_variants])
            r = next(x for x in nn_variants if x["variant"] == vname)
            regions_ready = [x for x in REGIONS if (region_cache(x) / "_coverage.csv").exists()]
            reg = c2.selectbox("Region", regions_ready, key="live_reg") if regions_ready else None
            if reg is None:
                st.warning("No cached data found (folder cache_lite/ is missing). Upload cache_lite/ with the app.")
                st.stop()
            cov = coverage(reg).set_index("dataid")
            regional = meta.get("regional_homes")
            elig = cov.index.tolist() if regional else [h for h in cov.index if all(
                not np.isnan(cov.loc[h].get(f"on%_{a}", np.nan)) for a in dc["appliances"])]
            if not elig:
                st.info("No home is available in this region.")
                st.stop()
            training = regional.get(reg, {}).get("train", []) if regional else (
                dc["homes"] if reg == dc["region"] else [])
            unseen = regional.get(reg, {}).get("unseen", []) if regional else (
                dc["unseen_homes"] if reg == (dc["unseen_region"] or dc["region"]) else [])
            hid = c2.selectbox("Home", elig, format_func=lambda h: f"{h}" + (
                " (train)" if h in training else " (unseen)" if h in unseen else " (new)"))
            start_day = pd.Timestamp(cov.loc[hid, "start"]).date()
            end_day = pd.Timestamp(cov.loc[hid, "end"]).date()
            day = c3.date_input("Day", value=max(start_day, end_day - pd.Timedelta(days=3)),
                                min_value=start_day, max_value=end_day, key=f"live_day_{reg}_{hid}")
            A, W = dc["appliances"], dc["window"]
            if regional:
                home = read_home(region_cache(reg) / f"{hid}.parquet", A, require_target=False)
                if home is None:
                    st.info("This home has no usable aggregate and appliance readings.")
                    st.stop()
                start = pd.Timestamp(day, tz="UTC")
                df = interpolate_short_gaps(home.loc[
                    (home.index >= start - pd.Timedelta(minutes=W - 1)) &
                    (home.index < start + pd.Timedelta(days=1))].copy())
            else:
                df, _ = cached_home(reg, hid, tuple(A), str(day - pd.Timedelta(days=1)), str(day))
                df = df.ffill().bfill()
            x = ((df["aggregate"].to_numpy() - meta["x_mean"]) / meta["x_std"]).astype(np.float32)
            if len(x) < W:
                st.warning("Not enough data for that day.")
            else:
                Xw = np.lib.stride_tricks.sliding_window_view(x, W)
                valid = np.isfinite(Xw).all(axis=1)
                if not valid.any():
                    st.info("No complete input windows for that day.")
                    st.stop()
                model = M.build(r["model"], W, len(A))
                model.load_state_dict(torch.load(RESULTS_DIR / sel_exp / f"{vname}_{'_'.join(A)}.pt",
                                                 map_location="cpu", weights_only=True))
                model.eval()
                with torch.no_grad():
                    lo, power = model(torch.from_numpy(np.ascontiguousarray(Xw[valid])))
                prob = np.full((len(Xw), len(A)), np.nan)
                pw = prob.copy()
                prob[valid] = torch.sigmoid(lo).numpy()
                pw[valid] = power.numpy()
                idx = df.index[W - 1:]
                mask = idx >= pd.Timestamp(day, tz="UTC")
                fig = go.Figure()
                fig.add_scatter(x=idx[mask], y=df["aggregate"].to_numpy()[W - 1:][mask], name="whole-house input",
                                line=dict(color="#999", width=1))
                cols = st.columns(len(A))
                for j, a in enumerate(A):
                    on = prob[:, j] > r["decision_thresholds"][a]
                    pred = np.where(on, np.maximum(pw[:, j] * meta["p_scale"][a], 0), 0)
                    pred[~valid] = np.nan
                    true = df[a].to_numpy()[W - 1:]
                    fig.add_scatter(x=idx[mask], y=true[mask], name=f"{a} actual",
                                    line=dict(color=PALETTE[j % len(PALETTE)], width=1.2))
                    fig.add_scatter(x=idx[mask], y=pred[mask], name=f"{a} predicted",
                                    line=dict(color=PALETTE[j % len(PALETTE)], width=1.2, dash="dot"))
                    scored = mask & valid & np.isfinite(true)
                    if scored.any():
                        yt = true[scored] > meta["thresholds_w"][a]
                        acc = (on[scored] == yt).mean()
                        e_t, e_p = true[scored].sum() / 60000, pred[scored].sum() / 60000
                        kpi(cols[j], a, f"{acc:.0%} acc")
                        cols[j].caption(f"energy actual {e_t:.2f} kWh · predicted {e_p:.2f} kWh")
                    else:
                        cols[j].caption(f"{a}: no measured target with a valid input window")
                fig.update_layout(height=460, yaxis_title="W", legend=dict(orientation="h"),
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

# ================================================================== REPORT
if page == "Report":
    st.header("Auto-generated report")
    if not blob:
        st.info("No experiment yet.")
    else:
        md = make_report(sel_exp)
        st.download_button("Download REPORT.md", md, f"NILM_report_{sel_exp}.md")
        p = RESULTS_DIR / sel_exp
        for f in ("metrics.csv", "regional_metrics.csv", "training_log.csv", "results.json"):
            if (p / f).exists():
                st.download_button(f"Download {f}", (p / f).read_bytes(), f"{sel_exp}_{f}", key=f)
        st.markdown(md)
