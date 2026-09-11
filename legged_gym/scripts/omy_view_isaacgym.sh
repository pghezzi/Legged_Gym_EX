#!/bin/sh
# Start a graphics-enabled IsaacGym shell, or execute the command passed as args.
# Usage: sh legged_gym/scripts/omy_view_isaacgym.sh [python -m ...]
# Optional: VIEWER_GPU=0 (default: host GPU 1).
set -eu

if [ -z "${DISPLAY:-}" ]; then
  echo "DISPLAY is unset. Run this from your graphical desktop terminal (X11/XWayland)." >&2
  exit 1
fi
for viewer_tool in docker xauth; do
  if ! command -v "$viewer_tool" >/dev/null 2>&1; then
    echo "Required host command missing: $viewer_tool" >&2
    exit 1
  fi
done

viewer_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$viewer_repo"
viewer_gpu=${VIEWER_GPU:-1}
case "$viewer_gpu" in
  ''|*[!0-9]*) echo "VIEWER_GPU must be a host GPU index, such as 0 or 1." >&2; exit 1 ;;
esac

# CUDA driver libraries alone do not register an NVIDIA Vulkan device. Use the
# host driver's manifest rather than guessing its library path/API version.
viewer_icd=
for viewer_candidate in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [ -r "$viewer_candidate" ]; then
    viewer_icd=$viewer_candidate
    break
  fi
done
if [ -z "$viewer_icd" ]; then
  echo "NVIDIA Vulkan manifest not found on the host; check the host graphics-driver installation." >&2
  exit 1
fi

# Share only this display's cookie, not the host's entire Xauthority file.
# FamilyWild allows the cookie to work with Docker's different hostname.
# No xhost permission changes are required.
viewer_cookie=$(xauth nlist "$DISPLAY")
if [ -z "$viewer_cookie" ]; then
  echo "No X11 cookie found for DISPLAY=$DISPLAY. Check your host XAUTHORITY setting." >&2
  exit 1
fi
umask 077
viewer_tmp=$(mktemp -d /tmp/leggedgym-viewer.XXXXXX)
cleanup() {
  rm -f -- "$viewer_tmp/Xauthority"
  rmdir -- "$viewer_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
touch "$viewer_tmp/Xauthority"
printf '%s\n' "$viewer_cookie" | sed 's/^..../ffff/' | xauth -f "$viewer_tmp/Xauthority" nmerge -
unset viewer_cookie

docker run --rm -it \
  --gpus "device=$viewer_gpu" \
  --ipc=host \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e SIMULATOR=isaacgym \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/viewer.Xauthority \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e TERRAIN="${TERRAIN:-gap}" \
  -e PARKOUR_AUX=0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$viewer_tmp/Xauthority:/tmp/viewer.Xauthority:ro" \
  -v "$viewer_icd:/usr/share/vulkan/icd.d/nvidia_icd.json:ro" \
  -v "$viewer_repo/legged_gym:/workspace/LeggedGym-Ex/legged_gym" \
  -v "$viewer_repo/rsl_rl:/workspace/LeggedGym-Ex/rsl_rl" \
  -v "$viewer_repo/logs:/workspace/LeggedGym-Ex/logs" \
  -w /workspace/LeggedGym-Ex \
  leggedgym-ex:isaacgym \
  bash -c '
    set -e
    export PATH="/workspace/LeggedGym-Ex/.venv/bin:$PATH"
    uv pip install --python /workspace/LeggedGym-Ex/.venv/bin/python python-dotenv
    if [ "$#" -gt 0 ]; then
      exec "$@"
    fi
    exec bash
  ' viewer "$@"
