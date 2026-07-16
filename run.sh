#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.runtime/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: benchmark runtime is missing; run ./bootstrap.sh first" >&2
  exit 2
fi

unset PYTHONPATH
export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"
export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"
cd "$ROOT"
exec "$PYTHON" "$ROOT/benchmark_cli.py" "$@"
