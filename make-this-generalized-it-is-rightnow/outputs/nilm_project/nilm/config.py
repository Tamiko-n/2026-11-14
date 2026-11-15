"""Central configuration for the NILM project.

Everything is path/region driven so the same pipeline generalises to any of the
supplied Pecan Street regions (or a combination of them).
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("NILM_DATA_ROOT", PROJECT_DIR.parent))
# full per-home cache if you ran ingest locally; otherwise the slim bundle shipped in the repo
CACHE_DIR = Path(os.environ.get("NILM_CACHE_DIR",
    PROJECT_DIR / ("cache_lite" if (PROJECT_DIR / "cache_lite" / "austin").is_dir()
                   and not (PROJECT_DIR / "cache").is_dir() else "cache")))
RESULTS_DIR = PROJECT_DIR / "results"


def region_cache(region, cache_dir=None):
    path = (CACHE_DIR if cache_dir is None else cache_dir) / region
    candidates = (
        path,
        DATA_ROOT / "nilm_project" / "cache" / region,
        PROJECT_DIR.parent.parent / "work" / "full_cache" / region,
        PROJECT_DIR.parent.parent / "work" / "cache" / region,
    )
    for candidate in candidates:
        if (candidate / "_coverage.csv").is_file() and any(candidate.glob("[0-9]*.parquet")):
            return candidate
    return path

# Glob patterns (relative to DATA_ROOT) for each region's raw 1-minute CSVs.
# Puerto Rico ships one folder per month/quantity: only *real power* is used.
REGIONS = {
    "austin": ["1minute_data_austin/**/1minute_data_austin.csv"],
    "california": ["1minute_data_california/**/1minute_data_california.csv"],
    "newyork": ["1minute_data_newyork*/**/1minute_data_newyork.csv"],
    "puertorico": ["pr_1min/**/pr_realpower_*_1min.csv"],
}

# Columns that are NOT appliances (never used as targets).
NON_APPLIANCE = {"dataid", "localminute", "grid", "solar", "solar2", "leg1v", "leg2v",
                 "battery1", "energy_storage_system", "energy_storage_system_l1",
                 "energy_storage_system_l2"}

SEED = 42
