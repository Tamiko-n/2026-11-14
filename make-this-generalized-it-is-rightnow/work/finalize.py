import json
import time
from nilm.config import RESULTS_DIR
from nilm.dashboard import make_dashboard
from nilm.report import make_report
from nilm.traces import make_traces
import pandas as pd
import torch
torch.set_num_threads(2)
folder = RESULTS_DIR / "multiregion"
deadline = time.monotonic() + 900
while True:
    blob = json.loads((folder / "results.json").read_text())
    if len(blob["results"]) == 8 and "total_time_s" in blob["meta"]:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError("Training has not completed")
    time.sleep(12)
make_traces("multiregion")
(folder / "REPORT.md").write_text(make_report("multiregion"), encoding="utf-8")
make_dashboard("multiregion")
metrics = pd.read_csv(folder / "regional_metrics.csv")
assert metrics.variant.nunique() == 8
assert set(metrics.region) == {"austin", "california", "newyork", "puertorico"}
assert (metrics.n_on == metrics.fn + metrics.tp).all()
assert (metrics.n_off == metrics.tn + metrics.fp).all()
assert not metrics.duplicated(["variant", "split", "region", "appliance"]).any()
traces = pd.read_csv(folder / "prediction_traces.csv")
assert traces.variant.nunique() == 8
assert traces.region.nunique() == 4
for region, homes in blob["meta"]["regional_homes"].items():
    assert not set(homes["train"]) & set(homes["unseen"])
    assert set(traces[traces.region.eq(region)].home).issubset(homes["unseen"])
print("Verified all eight models, disjoint holdouts, confusion counts, four regions and saved traces.")
