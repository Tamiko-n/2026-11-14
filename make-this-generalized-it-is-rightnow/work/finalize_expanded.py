import json
import time
import pandas as pd
import torch
from nilm.config import RESULTS_DIR
from nilm.models import MODELS
from nilm.dashboard import make_dashboard
from nilm.traces import make_traces
from nilm.report import make_report
from nilm.transfer import compare_training_regions

torch.set_num_threads(2)
folder = RESULTS_DIR / "expanded"
deadline = time.monotonic() + 3600
while True:
    path = folder / "results.json"
    if path.exists():
        blob = json.loads(path.read_text())
        if len(blob["results"]) == 14 and "total_time_s" in blob["meta"]:
            break
    if time.monotonic() >= deadline:
        raise TimeoutError("Expanded run did not finish")
    time.sleep(15)
meta = blob["meta"]
assert {r["model"] for r in blob["results"] if r["logic"] == "multi" and r["model"] in MODELS} == set(MODELS)
for region, homes in meta["regional_homes"].items():
    assert set(homes["train"]).isdisjoint(homes["unseen"])
    assert 2 * len(homes["train"]) >= meta["coverage_by_region"][region]["raw_homes"]
regional = pd.read_csv(folder / "regional_metrics.csv")
assert regional.variant.nunique() == 14 and regional.region.nunique() == 4
assert (regional.n_on + regional.n_off == regional[["tn","fp","fn","tp"]].sum(axis=1)).all()
compare_training_regions("expanded", "expanded_small", "training_size_comparison.csv")
make_traces("expanded")
traces = pd.read_csv(folder / "prediction_traces.csv")
assert traces.variant.nunique() == 14 and traces.region.nunique() == 4
(folder / "REPORT.md").write_text(make_report("expanded"), encoding="utf-8")
make_dashboard("expanded")
print("Verified training coverage, 14 variants, four regional comparisons and unseen-home traces.")
