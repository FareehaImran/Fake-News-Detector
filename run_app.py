import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable,
    "-m", "streamlit",
    "run", "src/streamlit_app.py",
    "--server.headless", "false",
    "--server.port", "8501"
])