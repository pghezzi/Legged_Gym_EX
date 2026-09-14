#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this checkout, including when launched inside Docker.
collection_repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$collection_repo"

# Set simulator
export SIMULATOR=isaacgym

# ---- Checkpoint variables ----
# Match Go2DepthWaqDistillCfg.distillation.teachers.
BASELINE_CKPT="$collection_repo/logs/go2_depth_waq_baseline/Sep09_05-49-24_dreamwaq_isaacgym/model_10000.pt"
GAP_CKPT="$collection_repo/logs/go2_depth_waq_fft_gap/Sep13_19-33-31_dreamwaq_isaacgym/model_50000.pt"
STAIRS_CKPT="$collection_repo/logs/go2_depth_waq_fft_all_stairs/Sep11_22-48-38_dreamwaq_isaacgym/model_50000.pt"
PIT_CKPT="$collection_repo/logs/go2_depth_waq_fft_pit/Sep13_20-20-37_dreamwaq_isaacgym/model_54500.pt"

MULTITASK_CKPTS=("$BASELINE_CKPT" "$GAP_CKPT" "$STAIRS_CKPT" "$PIT_CKPT")

# ---- Env vars shared by every run below ----
# the program will use this to create the env so please provide a working base
export TERRAIN=gap
export FINETUNE="$BASELINE_CKPT"

# ---- Run 1: single test on plane terrain ----
python -m legged_gym.scripts.play_exp_DO_NOT_TOUCH \
--task go2_depth_waq \
--no_depth_cam \
--hard_terrain_detector \
--save_depth_classifier_data \
--test_terrain plane \
--headless \
--num_envs 100 \
--multitask \
"${MULTITASK_CKPTS[@]}"

# ---- Run 2: multiterrain over seeds (42 69 100), viewer enabled ----
SEEDS=(42 69 100)
for seed in "${SEEDS[@]}"; do
    python -m legged_gym.scripts.play_exp_DO_NOT_TOUCH \
    --task go2_depth_waq \
    --no_depth_cam \
    --hard_terrain_detector \
    --save_depth_classifier_data \
    --multiterrain \
    --num_envs 100 \
    --headless \
    --multitask \
    "${MULTITASK_CKPTS[@]}" \
    --seed "$seed"
done

# ---- Run 3: multiterrain over seeds (47 92 220), with filtering ----
SEEDS=(47 92 220)
for seed in "${SEEDS[@]}"; do
    python -m legged_gym.scripts.play_exp_DO_NOT_TOUCH \
    --task go2_depth_waq \
    --no_depth_cam \
    --hard_terrain_detector \
    --save_depth_classifier_data \
    --multiterrain \
    --headless \
    --num_envs 100 \
    --filter_depth_classifier_data \
    --multitask \
    "${MULTITASK_CKPTS[@]}" \
    --seed "$seed"
done
