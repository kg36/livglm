#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
python_bin="$project_root/.venv/bin/python"
source_dir=${LIVGLM_NATIVE_MLX_SOURCE_DIR:-"$project_root/native/mlx-io-glm"}
wheel_dir="$project_root/native/dist"
overlay_dir="$project_root/native/testsite"
build_dir="$source_dir/build"
wheel_stage="$project_root/native/.dist-stage.$$"
overlay_stage="$project_root/native/.testsite-stage.$$"
revision="1b0930d12f6a3e84f88dfb4a66ae9380cb39d4cd"

required_native_smoke() {
  PYTHONPATH="$overlay_dir:$project_root/src" "$python_bin" - <<'PY'
import mlx.core as mx

required = (
    "_open_expert_safetensors_direct",
    "_expert_safetensors_direct_read_range_count",
    "_expert_ssd_direct_load_into",
    "_expert_ssd_direct_load_into_many",
    "_expert_ssd_route_plan",
    "_expert_ssd_markov_state_new",
    "_expert_ssd_route_cache_state_new",
    "_expert_ssd_route_cache_plan",
    "_expert_ssd_mxfp4_pair_qmv",
    "_expert_ssd_mxfp4_masked_qmv",
    "_expert_ssd_mxfp4_two_row_qmv",
    "_expert_ssd_two_row_gemv",
    "_expert_ssd_scalex_mxfp4_width2_pair_qmv",
    "_expert_ssd_scalex_mxfp4_width2_down_reduce",
    "_expert_ssd_wire_arrays",
    "_expert_ssd_unwire_arrays",
)
missing = [name for name in required if not hasattr(mx, name)]
if missing:
    raise SystemExit(f"mlx-io-glm overlay is missing: {missing}")
print("mlx-io-glm direct-to-slot overlay: valid")
PY
}

"$project_root/scripts/bootstrap_native_mlx.sh"

if [[ -f "$overlay_dir/.livglm-revision" ]] && \
   [[ "$(<"$overlay_dir/.livglm-revision")" == "$revision" ]] && \
   required_native_smoke; then
  print "mlx-io-glm overlay is already current."
  exit 0
fi

cleanup_staging() {
  rm -rf -- "$wheel_stage" "$overlay_stage"
}
trap cleanup_staging EXIT INT TERM

if [[ ! -f "$source_dir/setup.py" || ! -f "$source_dir/CMakeLists.txt" ]]; then
  print -u2 "refusing to clean unexpected native MLX source: $source_dir"
  exit 1
fi

rm -rf -- "$build_dir" "$wheel_stage" "$overlay_stage"
mkdir -p "$wheel_dir" "$wheel_stage" "$overlay_stage"
CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DMLX_METAL_DEBUG=OFF" \
CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8} \
  "$python_bin" -m pip wheel \
  "$source_dir" --no-deps --no-cache-dir -w "$wheel_stage"

wheel_path=""
for candidate in "$wheel_stage"/mlx-*.whl(N); do
  if [[ -z "$wheel_path" || "$candidate" -nt "$wheel_path" ]]; then
    wheel_path="$candidate"
  fi
done
if [[ -z "$wheel_path" ]]; then
  print -u2 "native MLX build did not produce a wheel"
  exit 1
fi

"$python_bin" -m pip install \
  --no-deps --upgrade --force-reinstall --target "$overlay_stage" \
  "$wheel_path"
print -r -- "$revision" > "$overlay_stage/.livglm-revision"

rm -rf -- "$overlay_dir"
mv "$overlay_stage" "$overlay_dir"
cp -f "$wheel_path" "$wheel_dir/${wheel_path:t}"
required_native_smoke

trap - EXIT INT TERM
