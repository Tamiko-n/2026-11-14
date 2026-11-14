import logging
import runpy
import warnings
from unittest.mock import patch

import streamlit as st
from streamlit.runtime.scriptrunner_utils import script_run_context

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
for page in ["Overview", "Data Explorer", "Train", "Results", "Comparison", "Generalisation", "Live Disaggregation", "Report"]:
    with patch.object(st.sidebar, "radio", return_value=page), patch.object(st.sidebar, "selectbox", return_value="multiregion"):
        runpy.run_path("outputs/nilm_project/app.py")
    print(page, "passed", flush=True)
