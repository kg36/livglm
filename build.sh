#!/bin/zsh
set -euo pipefail

project_root=${0:A:h}
python_command=${PYTHON313:-python3.13}

if ! command -v "$python_command" >/dev/null 2>&1; then
  print -u2 "Python 3.13 is required (set PYTHON313 to its executable)."
  exit 2
fi

if [[ ! -d "$project_root/.venv" ]]; then
  "$python_command" -m venv "$project_root/.venv"
fi

python_bin="$project_root/.venv/bin/python"
"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -e "${project_root}[dev]"
"$python_bin" -m pytest

print "Build and unit tests complete."
print "No full model run was performed. On the clean machine, try:"
print "  ./run.sh --preflight"
print "  ./run.sh --max-tokens 1 'Say hello in one word.'"
