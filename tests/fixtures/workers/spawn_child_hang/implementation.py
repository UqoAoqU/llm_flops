from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"]
)
marker = os.environ.get("BENCHMARK_ENGINE_CHILD_PID_FILE")
if marker:
    Path(marker).write_text(str(child.pid), encoding="utf-8")
time.sleep(60)
