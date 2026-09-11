#!/bin/sh
set -eu

# Resolve the repository relative to this script, on any host machine.
script_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$script_repo"

training_gpu=${TRAIN_GPU:-0}
case "$training_gpu" in
  ''|*[!0-9]*) echo "TRAIN_GPU must be a host GPU index, such as 0 or 1." >&2; exit 1 ;;
esac

docker run --rm -it \
  --gpus "device=$training_gpu" \
  --ipc=host \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e SIMULATOR=isaacgym \
  -v "$PWD/legged_gym:/workspace/LeggedGym-Ex/legged_gym" \
  -v "$PWD/rsl_rl:/workspace/LeggedGym-Ex/rsl_rl" \
  -v "$PWD/logs:/workspace/LeggedGym-Ex/logs" \
  -w /workspace/LeggedGym-Ex \
  leggedgym-ex:isaacgym \
  bash -c '
    set -e
    uv pip install --python /workspace/LeggedGym-Ex/.venv/bin/python python-dotenv
    export PATH="/workspace/LeggedGym-Ex/.venv/bin:$PATH"
    exec bash
  '
