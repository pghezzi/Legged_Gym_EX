#!/bin/sh
set -eu

cd /home/oyoungquist/Research/Genesis_Development/Legged_Gym_EX

docker run --rm -it \
  --gpus '"device=1"' \
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