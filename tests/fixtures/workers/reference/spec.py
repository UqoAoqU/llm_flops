from __future__ import annotations

import os
from pathlib import Path


marker = os.environ.get("BENCHMARK_ENGINE_IMPORT_ORDER")
if marker:
    with Path(marker).open("a", encoding="utf-8") as stream:
        stream.write("spec\n")


class _Spec:
    pass


spec = _Spec()
