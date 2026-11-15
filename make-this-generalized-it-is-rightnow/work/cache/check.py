import logging
from pathlib import Path
from streamlit.testing.v1 import AppTest
logging.getLogger("streamlit").setLevel(logging.ERROR)
a=AppTest.from_file(str(Path("outputs/nilm_project/app.py").resolve()), default_timeout=25).run()
print("Overview", [e.message for e in a.exception], flush=True)
for page in ["Train", "Results", "Comparison", "Generalisation", "Live Disaggregation", "Report"]:
    a.sidebar.radio[0].set_value(page).run()
    print(page, [e.message for e in a.exception], flush=True)
    assert not a.exception

