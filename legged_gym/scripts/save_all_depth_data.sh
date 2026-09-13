#!/usr/bin/env bash
set -euo pipefail

# Set simulator
export SIMULATOR=isaacgym

# ---- Checkpoint variables ----
BASELINE_CKPT=/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_baseline/Aug09_02-33-11_dreamwaq_isaacgym/model_7000.pt
GAP_CKPT=/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_gap/Aug12_17-01-51_dreamwaq_isaacgym/model_47000.pt
STAIRS_CKPT=/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_all_stairs/Aug14_14-23-53_dreamwaq_isaacgym/model_47000.pt
PIT_CKPT=/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_pit/Aug29_00-30-31_dreamwaq_isaacgym/model_67000.pt

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

# ---- Run 2: multiterrain over seeds (42 69 100) ----
SEEDS=(42 69 100)
for seed in "${SEEDS[@]}"; do
    python -m legged_gym.scripts.play_exp_DO_NOT_TOUCH \
    --task go2_depth_waq \
    --no_depth_cam \
    --hard_terrain_detector \
    --save_depth_classifier_data \
    --multiterrain \
    --headless \
    --num_envs 100 \
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