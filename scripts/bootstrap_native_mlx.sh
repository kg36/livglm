#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
source_dir=${LIVGLM_NATIVE_MLX_SOURCE_DIR:-"$project_root/native/mlx-io-glm"}
repository="https://github.com/kg36/mlx-io-glm.git"
revision="1b0930d12f6a3e84f88dfb4a66ae9380cb39d4cd"

if [[ -e "$source_dir" ]]; then
  if ! git -C "$source_dir" rev-parse --git-dir >/dev/null 2>&1; then
    print -u2 "native MLX path exists but is not a Git checkout: $source_dir"
    exit 1
  fi
else
  mkdir -p "$source_dir"
  git -C "$source_dir" init
  git -C "$source_dir" remote add origin "$repository"
fi

if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]]; then
  print -u2 "refusing to change a dirty native MLX checkout: $source_dir"
  exit 1
fi

remote_url=$(git -C "$source_dir" remote get-url origin 2>/dev/null || true)
if [[ "$remote_url" != "$repository" && \
      "$remote_url" != "git@github.com:kg36/mlx-io-glm.git" ]]; then
  print -u2 "native MLX origin points somewhere unexpected: $remote_url"
  exit 1
fi

head_revision=$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)
if [[ "$head_revision" != "$revision" ]]; then
  git -C "$source_dir" fetch --depth 1 origin "$revision"
fi
git -C "$source_dir" cat-file -e "$revision^{commit}"
git -C "$source_dir" checkout --detach "$revision"

resolved=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$resolved" != "$revision" ]]; then
  print -u2 "native MLX revision mismatch: expected $revision, got $resolved"
  exit 1
fi

print "native MLX source: $source_dir"
print "repository: $repository"
print "revision: $resolved"
