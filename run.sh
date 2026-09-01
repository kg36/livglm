#!/bin/zsh
set -euo pipefail

project_root=${0:A:h}
python_bin="$project_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  print -u2 "LivGLM is not built. Run ./build.sh first."
  exit 2
fi

native_overlay="$project_root/native/testsite"
if [[ ! -d "$native_overlay" ]]; then
  print -u2 "mlx-io-glm overlay is missing. Run ./build.sh first."
  exit 2
fi

for argument in "$@"; do
  if [[ "$argument" == "--trace" || "$argument" == --trace=* ]]; then
    export MLX_PROFILE_METAL_COUNTERS=1
    break
  fi
done

export PYTHONPATH="$native_overlay:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m glm53flash.chat_cli "$@"
