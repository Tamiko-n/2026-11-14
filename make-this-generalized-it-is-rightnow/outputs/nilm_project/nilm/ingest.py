"""Step 1 - read the huge regional CSVs once (chunked) and cache one parquet per home.

Also builds a *coverage table* (home x appliance -> ON fraction) so homes and
appliances can be chosen with evidence instead of guessing.

Usage:  python -m nilm.ingest --region austin
"""
import argparse
import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .config import DATA_ROOT, CACHE_DIR, REGIONS, NON_APPLIANCE

ON_PROBE_W = 30.0  # only for the coverage survey; real thresholds are learned from training data


def region_files(region):
    files = []
    for pat in REGIONS[region]:
        files += glob.glob(str(DATA_ROOT / pat), recursive=True)
    return sorted({f for f in files if Path(f).is_file()})


def ingest_region(region, chunksize=200_000, columns=None):
    out_dir = CACHE_DIR / region
    out_dir.mkdir(parents=True, exist_ok=True)
    files = region_files(region)
    if not files:
        raise FileNotFoundError(f"No CSV files for region {region}")
    with TemporaryDirectory(prefix="_parts_", dir=out_dir) as temp:
        return _ingest(region, files, out_dir, Path(temp), chunksize, columns)


def _ingest(region, files, out_dir, tmp, chunksize, columns):
    k = 0
    usecols = (lambda name: name in columns) if columns is not None else None
    for f in files:
        print(f"[ingest] {f}")
        for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False, usecols=usecols):
            chunk = chunk.loc[:, [c for c in chunk.columns if not c.startswith("Unnamed")]]
            num = chunk.columns.difference(["dataid", "localminute"])
            chunk[num] = chunk[num].astype("float32")
            for hid, g in chunk.groupby("dataid"):   # stream each home's slice to disk (low RAM)
                (tmp / str(int(hid))).mkdir(exist_ok=True)
                g.to_parquet(tmp / str(int(hid)) / f"{k:05d}.parquet", index=False)
            k += 1
            print(f"   chunk {k}: {len(chunk):,} rows")
    rows = []
    for hdir in sorted(tmp.iterdir()):
        hid = int(hdir.name)
        lst = [pd.read_parquet(p) for p in sorted(hdir.glob("*.parquet"))]
        df = pd.concat(lst, ignore_index=True).dropna(axis=1, how="all")
        df["localminute"] = pd.to_datetime(df["localminute"], utc=True)
        df = df.sort_values("localminute").drop_duplicates("localminute")
        df.to_parquet(out_dir / f"{hid}.parquet", index=False)
        rows.append(coverage_row(region, hid, df))
        print(f"[ingest] home {hid}: {len(df):,} rows saved")
        for p in hdir.glob("*.parquet"):
            p.unlink()
        hdir.rmdir()
        del df, lst
    cov = pd.DataFrame(rows)
    cov.to_csv(out_dir / "_coverage.csv", index=False)
    return cov


def coverage_row(region, hid, df):
    r = {"region": region, "dataid": hid, "rows": len(df),
         "start": df["localminute"].min(), "end": df["localminute"].max(),
         "has_grid": "grid" in df, "has_solar": "solar" in df}
    for c in df.columns:
        if c in NON_APPLIANCE:
            continue
        v = df[c].to_numpy() * 1000.0
        valid = ~np.isnan(v)
        if valid.mean() < 0.5:
            continue
        r[f"on%_{c}"] = round(float((v[valid] > ON_PROBE_W).mean() * 100), 2)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=[*REGIONS, "all"], required=True)
    ap.add_argument("--targets", nargs="+", help="limit the cache to these appliance columns and whole-home power")
    a = ap.parse_args()
    columns = {"dataid", "localminute", "grid", "solar", "solar2", *a.targets} if a.targets else None
    for region in REGIONS if a.region == "all" else [a.region]:
        cov = ingest_region(region, columns=columns)
        print(f"{region}: {len(cov)} homes cached")
