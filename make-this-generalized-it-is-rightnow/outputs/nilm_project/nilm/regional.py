"""Pooled regional datasets with chronological and whole-home holdouts."""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from .config import CACHE_DIR, REGIONS, region_cache
from .data import learn_threshold


@dataclass
class RegionalConfig:
    regions: list = field(default_factory=lambda: list(REGIONS))
    appliances: list = field(default_factory=lambda: ["air1", "furnace1", "drye1", "refrigerator1"])
    window: int = 60
    train_frac: float = 0.70
    val_frac: float = 0.15
    train_per_home: int = 2500
    eval_per_home: int = 1200
    seed: int = 42
    minimum_window_coverage: float = 0.0

    def to_dict(self):
        return asdict(self)


def interpolate_short_gaps(frame):
    for column in frame:
        series = frame[column]
        missing = series.isna()
        runs = missing.ne(missing.shift()).cumsum()
        short = missing.groupby(runs).transform("sum").le(5)
        interpolated = series.interpolate(limit_area="inside")
        frame[column] = series.where(~(missing & short), interpolated)
    return frame


def read_home(path, appliances, require_target=True):
    available = set(pq.ParquetFile(path).schema.names)
    targets = available.intersection(appliances)
    if "grid" not in available or (require_target and not targets):
        return None
    columns = ["localminute", "grid", *sorted(available.intersection({"solar", "solar2"})), *sorted(targets)]
    frame = pd.read_parquet(path, columns=columns)
    frame.index = pd.to_datetime(frame.pop("localminute"), utc=True).dt.floor("min")
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated()]
    solar = frame.reindex(columns=["solar", "solar2"]).fillna(0).sum(axis=1)
    watts = frame.reindex(columns=appliances)
    watts.insert(0, "aggregate", frame.grid + solar)
    return (watts * 1000).replace([np.inf, -np.inf], np.nan).clip(lower=0).asfreq("min")


def windows(frame, width, limit, coverage=0.0):
    values = interpolate_short_gaps(frame.copy()).to_numpy(np.float32)
    if len(values) < width:
        return None
    history = sliding_window_view(values[:, 0], width)
    targets = values[width - 1:, 1:]
    valid = np.flatnonzero(np.isfinite(history).all(axis=1) & np.isfinite(targets).any(axis=1))
    if len(valid) < 2:
        return None
    available = len(valid)
    if coverage:
        limit = max(limit, int(np.ceil(available * coverage)))
    if limit and len(valid) > limit:
        valid = valid[np.linspace(0, len(valid) - 1, limit, dtype=int)]
    part = {"X": history[valid], "P": targets[valid], "T": frame.index.values[valid + width - 1]}
    return part, available


def build_regional(cfg, cache_dir=None, train_regions=None, log=print):
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    train_regions = set(cfg.regions if train_regions is None else train_regions)
    if cfg.window < 2 or not train_regions or not train_regions.issubset(cfg.regions):
        raise ValueError("Use a window of at least two minutes and one or more selected training regions")
    if cfg.train_per_home < 0 or cfg.eval_per_home < 0 or not 0 <= cfg.minimum_window_coverage <= 1:
        raise ValueError("Use nonnegative window caps and coverage between 0 and 1")
    if not 0 < cfg.train_frac < cfg.train_frac + cfg.val_frac < 1:
        raise ValueError("Training, validation and future test fractions must all be positive")
    pieces = {name: [] for name in ("train", "val", "test", "unseen")}
    homes, cleaning, split_bounds, coverage_by_region = {}, {}, {}, {}
    for region in cfg.regions:
        region_dir = region_cache(region, cache_dir)
        if not (region_dir / "_coverage.csv").is_file():
            raise FileNotFoundError(f"Run python -m nilm.ingest --region {region} first")
        paths = sorted(region_dir.glob("[0-9]*.parquet"))
        eligible = {}
        skipped = []
        for path in paths:
            hid = int(path.stem)
            frame = read_home(path, cfg.appliances)
            if frame is None or len(frame) < 7 * 1440:
                skipped.append(hid)
                continue
            cut1 = int(len(frame) * cfg.train_frac)
            cut2 = int(len(frame) * (cfg.train_frac + cfg.val_frac))
            partitions = {"train": frame.iloc[:cut1], "val": frame.iloc[cut1:cut2], "test": frame.iloc[cut2:]}
            parts = {
                name: windows(part, cfg.window, cfg.train_per_home if name == "train" else cfg.eval_per_home,
                              cfg.minimum_window_coverage if name == "train" else 0)
                for name, part in partitions.items()
            }
            if any(part is None for part in parts.values()):
                skipped.append(hid)
                continue
            key = f"{region}/{hid}"
            cleaning[key] = {"grid_minutes": len(frame), "missing_aggregate": int(frame["aggregate"].isna().sum()),
                             "measured_targets": [a for a in cfg.appliances if frame[a].notna().any()]}
            split_bounds[key] = {name: [str(part.index[0]), str(part.index[-1])]
                                 for name, part in partitions.items()}
            eligible[hid] = (parts, path)
        if len(eligible) < 2:
            raise ValueError(f"{region} needs at least two homes with usable data in each temporal partition")
        ordered = np.random.default_rng(cfg.seed).permutation(sorted(eligible))
        unseen = set(ordered[:max(1, round(0.2 * len(ordered)))].tolist())
        homes[region] = {"train": [], "unseen": sorted(unseen), "excluded": skipped}
        coverage_by_region[region] = {"raw_homes": len(paths), "train_windows": 0, "available_train_windows": 0}
        for hid, (parts, path) in eligible.items():
            if hid in unseen:
                frame = read_home(path, cfg.appliances)
                whole_home = windows(frame, cfg.window, cfg.eval_per_home)
                if whole_home is None:
                    raise ValueError(f"{region}/{hid} has no usable unseen windows")
                parts = {"unseen": whole_home}
                split_bounds[f"{region}/{hid}"] = {"unseen": [str(frame.index[0]), str(frame.index[-1])]}
            elif region in train_regions:
                homes[region]["train"].append(hid)
            else:
                parts = {"test": parts["test"]}
            for name, (part, available) in parts.items():
                n = len(part["X"])
                if name == "train":
                    coverage_by_region[region]["train_windows"] += n
                    coverage_by_region[region]["available_train_windows"] += available
                pieces[name].append(dict(part, H=np.full(n, hid), R=np.full(n, region)))
        log(f"{region}: {len(homes[region]['train'])} training, {len(unseen)} unseen, {len(skipped)} excluded homes")

    splits = {name: {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
              for name, parts in pieces.items() if parts}
    train = splits["train"]
    mean, std = float(train["X"].mean(dtype=np.float64)), max(float(train["X"].std(dtype=np.float64)), 1e-6)
    thresholds, scales = {}, {}
    for j, appliance in enumerate(cfg.appliances):
        watts = train["P"][:, j]
        thresholds[appliance] = learn_threshold(watts)
        if not (np.any(watts > thresholds[appliance]) and np.any(watts <= thresholds[appliance])):
            raise ValueError(f"No usable ON/OFF training examples for {appliance}")
        scales[appliance] = max(float(np.nanpercentile(watts, 99.5)), thresholds[appliance])
    limits = np.array(list(thresholds.values()), np.float32)
    counts = {}
    for name, part in splits.items():
        part["X"] -= mean
        part["X"] /= std
        part["Y"] = np.where(np.isfinite(part["P"]), part["P"] > limits, np.nan).astype(np.float32)
        counts[name] = {a: {"ON": int((part["Y"][:, j] == 1).sum()),
                            "OFF": int((part["Y"][:, j] == 0).sum())}
                        for j, a in enumerate(cfg.appliances)}
        log(f"{name}: {len(part['X']):,} windows")
    return {"splits": splits, "appliances": cfg.appliances, "thresholds": thresholds,
            "p_scale": scales, "x_mean": mean, "x_std": std, "class_counts": counts,
            "cleaning": cleaning, "split_bounds": split_bounds, "regional_homes": homes,
            "train_regions": sorted(train_regions), "coverage_by_region": coverage_by_region}

