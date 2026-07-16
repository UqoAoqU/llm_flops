#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.runtime/venv"

unset PYTHONPATH
export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"
export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: benchmark runtime is missing; run ./bootstrap.sh first" >&2
  exit 2
fi

exec "$VENV/bin/python" -m benchmark_engine "$@"
