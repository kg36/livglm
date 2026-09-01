#!/bin/zsh
set -euo pipefail

project_root=${0:A:h}
python_bin="$project_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  print -u2 "LivGLM is not built. Run ./build.sh first."
  exit 2
fi

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m glm53flash.chat_cli "$@"
