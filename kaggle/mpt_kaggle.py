"""Launch the MoneyPrinterTurbo Streamlit UI as a separate Kaggle process."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
candidates = [ROOT / "webui" / "Main.py", ROOT / "webui" / "main.py", ROOT / "webui" / "Home.py"]
entry = next((p for p in candidates if p.is_file()), None)
if entry is None:
    raise FileNotFoundError("Could not locate webui/Main.py, webui/main.py, or webui/Home.py.")
cmd = [
    sys.executable, "-m", "streamlit", "run", str(entry),
    "--server.address", os.environ.get("MPT_STREAMLIT_HOST", "0.0.0.0"),
    "--server.port", os.environ.get("MPT_STREAMLIT_PORT", "8501"),
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
]
print("[MPT] Starting Streamlit UI on port 8501", flush=True)
raise SystemExit(subprocess.call(cmd))
