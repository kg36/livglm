#!/bin/zsh
set -euo pipefail
ROOT=${0:A:h:h}
exec /Users/kumargaurav/Documents/LLMs/DSv4Flash/main/.venv/bin/python "$ROOT/scripts/preflight.py" "$@"
