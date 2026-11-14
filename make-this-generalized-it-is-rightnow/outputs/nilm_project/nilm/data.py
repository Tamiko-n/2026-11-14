"""Steps 2-6: clean, label, split (time order, no leakage) and window the data.

Design rules enforced here (from the CCP brief):
  * X = whole-house power ONLY (grid + solar = household consumption).
  * Appliance power is only ever used as a target y.
  * Splits are in time order per home (70/15/15) plus optional completely unseen homes.
  * ON thresholds and the input scaler are fitted on TRAINING data only.
  * Windows never cross a split boundary or a long data gap.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .config import region_cache

MAX_INTERP_GAP = 5          # minutes of missing data we are willing to interpolate
MIN_ON_W = 10.0             # floor for any learned ON threshold (sensor noise level)


@dataclass
class DataConfig:
    region: str = "austin"
    homes: list = field(default_factory=lambda: [661, 2335, 2361, 3456, 4767, 8565])
    unseen_homes: list = field(default_factory=lambda: [4373, 7536])
    unseen_region: str = ""      # "" = same region; else test unseen homes from another region
    unseen_start: str = ""       # "" = same dates as training homes
    unseen_end: str = ""
    appliances: list = field(default_factory=lambda: ["air1", "furnace1", "drye1", "refrigerator1"])
    start: str = "2018-05-01"
    end: str = "2018-08-31"
    window: int = 60
    train_frac: float = 0.70
    val_frac: float = 0.15
    train_stride: int = 2       # subsample training windows (CPU budget); val/test use stride 1
    threshold_method: str = "kmeans"   # "kmeans" (data-driven) or "fixed"
    fixed_threshold_w: float = 50.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- loading / cleaning
def load_home(region, hid, appliances, start=None, end=None):
    """Return a continuous 1-minute frame: 'aggregate' + appliance columns, in watts."""
    df = pd.read_parquet(region_cache(region) / f"{hid}.parquet")
    df = df.set_index("localminute").sort_index()
    df = df[~df.index.duplicated()]
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if df.empty:
        raise ValueError(f"home {hid} has no readings in the selected period")
    missing = [a for a in appliances if a not in df.columns]
    if missing:
        raise ValueError(f"home {hid} has no column(s) {missing}")
    if "grid" not in df:
        raise ValueError(f"home {hid} has no 'grid' column")
    solar = df[[c for c in ("solar", "solar2") if c in df]].fillna(0).sum(axis=1)
    out = pd.DataFrame(index=df.index)
    out["aggregate"] = (df["grid"] + solar) * 1000.0     # kW -> W, household consumption
    for a in appliances:
        out[a] = df[a] * 1000.0
    # continuous minute grid, short gaps interpolated, long gaps left NaN (windows skip them)
    full = pd.date_range(out.index.min(), out.index.max(), freq="1min")
    report = {"rows_raw": len(out), "rows_grid": len(full)}
    out = out.reindex(full)
    report["missing_before"] = int(out["aggregate"].isna().sum())
    out = out.interpolate(limit=MAX_INTERP_GAP, limit_area="inside")
    report["missing_after"] = int(out["aggregate"].isna().sum())
    out = out.clip(lower=0)                               # negative sensor noise -> 0
    return out.astype("float32"), report


# ---------------------------------------------------------------- thresholds
def learn_threshold(power_w, method="kmeans", fixed=50.0):
    """Pick an ON/OFF threshold from TRAINING appliance power only.

    kmeans: 1-D 2-cluster k-means on log power; threshold = midpoint of the
    two centroids (in log space), floored at MIN_ON_W.  Works for 100 W fridges
    and 3 kW AC units alike, which is what makes the pipeline generalise.
    """
    p = power_w[~np.isnan(power_w)]
    if method == "fixed" or len(p) == 0:
        return float(fixed)
    x = np.log1p(p)
    # extreme-percentile init so rare appliances (dryer ON ~2 %) still get an ON cluster
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.9)
    if hi - lo < 1e-3:
        return float(max(fixed, MIN_ON_W))
    c = np.array([lo, hi])
    for _ in range(100):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        new = np.array([x[lab == k].mean() if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    thr = float(np.expm1(c.mean()))
    return max(thr, MIN_ON_W)


# ---------------------------------------------------------------- dataset assembly
class Segments:
    """Holds contiguous arrays per (home, split) and valid window-end indices."""

    def __init__(self, window):
        self.window = window
        self.x, self.y, self.t, self.home = [], [], [], []
        self.ends = []  # list of (segment_id, end_idx)

    def add(self, home, agg, app, times, stride):
        sid = len(self.x)
        self.x.append(agg)
        self.y.append(app)
        self.t.append(times)
        self.home.append(home)
        ok = ~np.isnan(agg) & ~np.isnan(app).any(1)
        # window ending at e is valid if all W points [e-W+1, e] are ok
        cs = np.concatenate([[0], np.cumsum(~ok)])
        e = np.arange(self.window - 1, len(agg))
        valid = (cs[e + 1] - cs[e + 1 - self.window]) == 0
        e = e[valid][::stride]
        self.ends += [(sid, i) for i in e]

    def arrays(self, x_mean, x_std):
        """Materialise windows: X (N, W) scaled, P (N, A) watts, plus meta."""
        W = self.window
        n = len(self.ends)
        A = self.y[0].shape[1] if self.y else 0
        X = np.empty((n, W), np.float32)
        P = np.empty((n, A), np.float32)
        T = np.empty(n, dtype="datetime64[ns]")
        H = np.empty(n, np.int64)
        for k, (s, e) in enumerate(self.ends):
            X[k] = self.x[s][e - W + 1:e + 1]
            P[k] = self.y[s][e]
            T[k] = self.t[s][e]
            H[k] = self.home[s]
        return (X - x_mean) / x_std, P, T, H


def build_dataset(cfg: DataConfig, log=print):
    A = cfg.appliances
    W = cfg.window
    seg = {k: Segments(W) for k in ("train", "val", "test", "unseen")}
    cleaning, frames = {}, {}
    for hid in list(cfg.homes) + list(cfg.unseen_homes):
        if hid in cfg.unseen_homes:
            reg, st, en = cfg.unseen_region or cfg.region, cfg.unseen_start or cfg.start, cfg.unseen_end or cfg.end
        else:
            reg, st, en = cfg.region, cfg.start, cfg.end
        df, rep = load_home(reg, hid, A, st, en)
        cleaning[hid] = rep
        frames[hid] = df
        log(f"home {hid}: {len(df):,} minutes, missing {rep['missing_before']} -> {rep['missing_after']}")

    # --- time-ordered split per seen home
    split_bounds = {}
    train_power = {a: [] for a in A}
    train_agg = []
    for hid in cfg.homes:
        df = frames[hid]
        n = len(df)
        i1, i2 = int(n * cfg.train_frac), int(n * (cfg.train_frac + cfg.val_frac))
        split_bounds[hid] = {"train": [str(df.index[0]), str(df.index[i1 - 1])],
                             "val": [str(df.index[i1]), str(df.index[i2 - 1])],
                             "test": [str(df.index[i2]), str(df.index[-1])]}
        for name, sl in (("train", slice(0, i1)), ("val", slice(i1, i2)), ("test", slice(i2, n))):
            part = df.iloc[sl]
            seg[name].add(hid, part["aggregate"].to_numpy(), part[A].to_numpy(),
                          part.index.values, cfg.train_stride if name == "train" else 1)
            if name == "train":
                train_agg.append(part["aggregate"].to_numpy())
                for a in A:
                    train_power[a].append(part[a].to_numpy())
    for hid in cfg.unseen_homes:
        df = frames[hid]
        split_bounds[hid] = {"unseen_test": [str(df.index[0]), str(df.index[-1])]}
        seg["unseen"].add(hid, df["aggregate"].to_numpy(), df[A].to_numpy(), df.index.values, 1)

    # --- fit thresholds + scaler on TRAIN only
    thresholds = {a: learn_threshold(np.concatenate(train_power[a]), cfg.threshold_method,
                                     cfg.fixed_threshold_w) for a in A}
    agg = np.concatenate(train_agg)
    x_mean, x_std = float(np.nanmean(agg)), float(np.nanstd(agg) + 1e-6)
    p_scale = {a: float(np.nanpercentile(np.concatenate(train_power[a]), 99.5) + 1e-6) for a in A}

    out = {"appliances": A, "thresholds": thresholds, "x_mean": x_mean, "x_std": x_std,
           "p_scale": p_scale, "cleaning": cleaning, "split_bounds": split_bounds, "splits": {}}
    thr = np.array([thresholds[a] for a in A], np.float32)
    counts = {}
    for name, s in seg.items():
        if not s.ends:
            continue
        X, P, T, H = s.arrays(x_mean, x_std)
        Y = (P > thr).astype(np.float32)
        out["splits"][name] = {"X": X, "P": P, "Y": Y, "T": T, "H": H}
        counts[name] = {a: {"ON": int(Y[:, j].sum()), "OFF": int(len(Y) - Y[:, j].sum())}
                        for j, a in enumerate(A)}
        log(f"{name}: {len(X):,} windows | " +
            ", ".join(f"{a} ON {counts[name][a]['ON']:,}/OFF {counts[name][a]['OFF']:,}" for a in A))
    out["class_counts"] = counts
    log("thresholds (W): " + ", ".join(f"{a}={t:.1f}" for a, t in thresholds.items()))
    return out
