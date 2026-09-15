from __future__ import annotations

from legged_gym import LEGGED_GYM_ROOT_DIR

import os
from pathlib import Path

from dotenv import load_dotenv

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import (
    Go2DepthWaqCfg,
    Go2DepthWaqCfgPPO,
)

# Load environment variables from:
#
# Legged_Gym_EX/.env
#
# override=False means values already exported in the shell take priority
# over values written in the .env file.
load_dotenv(
    dotenv_path= Path(LEGGED_GYM_ROOT_DIR) / ".env",
    override=False,
)

import warnings


def checkpoint_path(path: str) -> str:
    """Resolve relative checkpoints from the repository root, not the launch directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(LEGGED_GYM_ROOT_DIR) / path
    return str(path.resolve())


def required_env(name: str) -> str:
    """Read a required environment variable.

    Warns when the variable has not been provided through the shell or the
    repository's local .env file.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        warnings.warn(
            f"Required environment variable {name!r} is not set.\n"
            f"Create the local file:\n"
            f"  {Path(LEGGED_GYM_ROOT_DIR) / '.env'}\n"
            f"using the committed .env.example file.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ""

    return value.strip()


def optional_int_env(name: str, default: int) -> int:
    """Read an optional integer environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return int(default)

    try:
        return int(raw_value)
    except ValueError:
        warnings.warn(
            f"Environment variable {name!r} must be an integer, "
            f"but received {raw_value!r}. Using default {default!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return int(default)



class Go2DepthWaqDistillCfg(Go2DepthWaqCfg):
    """Mixed rough/stairs/gap/pit terrain with fixed depth-policy teachers."""

    class viewer(Go2DepthWaqCfg.viewer):
        # Rendering only the two camera environments keeps visual smoke tests
        # lightweight. CLI runs with one environment clamp this list to [0].
        rendered_envs_idx = [0, 1]

    class terrain(Go2DepthWaqCfg.terrain):
        # Proportions follow TERRAIN_KEYS. With 12 curriculum columns, these
        # realize 3 rough, 5 stairs (both directions), 2 gap, and 2 pit columns.
        curriculum = True
        selected = False
        custom_selected = False
        num_cols = 12
        terrain_proportions = [
            0.0,  # slope
            0.20,  # random_uniform
            0.20,  # stairs
            0.20,  # upwards_stairs
            0.0,  # discrete_obstacles
            0.0,  # stepping_stones
            0.20,  # gap
            0.20,  # pit
            0.0,  # multiple_high_platforms
            0.0,  # high_platform_gaps
            0.0,
        ]

    class rewards(Go2DepthWaqCfg.rewards):
        class obstacle_progress(Go2DepthWaqCfg.rewards.obstacle_progress):
            # This helper supports dedicated obstacle terrains, not this mixture.
            enabled = False

        # Gap and stairs teachers were trained with the same reward scales.
        # Define them explicitly because the parent config selects its scales
        # at import time from TERRAIN, whose default is random_uniform.
        class scales:
            feet_prolonged_air_time = Go2DepthWaqCfg.rewards.scales.feet_prolonged_air_time
            base_height = -1.0
            torque_limits = -0.001
            dof_pos_limits = -2.0
            collision = -10.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_power = -2e-05
            dof_acc = -2e-07
            action_rate = -0.01
            action_smoothness = -0.01
            hip_pos = -0.15
            foot_clearance_terrain_aware = 0.7
            feet_stumble = -1.0
            feet_near_edge = -1.0
            feet_air_time = 0.6

    class distillation:
        teacher_actor_critic = "ActorCriticDreamWaQDepthLora"
        # ------------------------------------------------------------------
        # Shared LoRA teacher defaults
        # ------------------------------------------------------------------
        #
        # Only LoRA teachers use this initialization checkpoint. Ordinary
        # ActorCriticDreamWaQDepth teachers load their own full checkpoint.
        # Relative .env overrides are also resolved from the repository root.
        base_model = checkpoint_path(os.getenv("DISTILL_BASE_MODEL") or
            "logs/go2_depth_waq_baseline/Sep09_05-49-24_dreamwaq_isaacgym/model_10000.pt")

        # All LoRA components use this rank unless a teacher dictionary
        # overrides "rank" or a component-specific rank.
        lora_rank = optional_int_env(
            "DISTILL_LORA_RANK",
            default=8,
        )

        # ------------------------------------------------------------------
        # Full-checkpoint teachers (LoRA can also be selected per dictionary)
        # ------------------------------------------------------------------
        #
        # List order defines the numeric teacher ID:
        #
        #   teachers[0] -> teacher ID 0 -> gap
        #   teachers[1] -> teacher ID 1 -> stairs
        #   teachers[2] -> teacher ID 2 -> pit
        #   teachers[3] -> teacher ID 3 -> rough baseline
        #
        # Paths below are relative to LEGGED_GYM_ROOT_DIR. Absolute paths are
        # also accepted by checkpoint_path(). Match the class to the checkpoint.
        teachers = [
            {
                "name": "gap",
                "checkpoint": checkpoint_path("logs/go2_depth_waq_fft_gap/Sep13_19-33-31_dreamwaq_isaacgym/model_50000.pt"),
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
            {
                "name": "stairs",
                "checkpoint": checkpoint_path("logs/go2_depth_waq_fft_all_stairs/Sep11_22-48-38_dreamwaq_isaacgym/model_50000.pt"),
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
            {
                "name": "pit",
                "checkpoint": checkpoint_path("logs/go2_depth_waq_fft_pit/Sep13_20-20-37_dreamwaq_isaacgym/model_54500.pt"),
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
            {
                "name": "rough",
                "checkpoint": checkpoint_path("logs/go2_depth_waq_baseline/Sep09_05-49-24_dreamwaq_isaacgym/model_10000.pt"),
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
        ]

        # ------------------------------------------------------------------
        # Terrain column -> teacher mapping
        # ------------------------------------------------------------------
        #
        # Index:
        #   Simulator terrain column ID (not a TERRAIN_KEYS index)
        #
        # Value:
        #   index into the teachers list above
        #
        # Columns 0-2: rough; 3-7: stairs; 8-9: gap; 10-11: pit.
        terrain_type_to_teacher = [3] * 3 + [1] * 5 + [0] * 2 + [2] * 2
        # ------------------------------------------------------------------
        # Pure imitation-learning settings
        # ------------------------------------------------------------------
        #
        # Supported targets in the supplied PPO_WAQ_Distill scaffold:
        #
        #   "l1"      -> sum absolute action error per sample
        #   "mse"     -> mean squared action error per sample
        #   "mse_sum" -> summed squared action error per sample
        #   "l2"      -> Euclidean action-vector distance
        distill_target = "l1"

        distillation_loss_coef = 1.0

        learning_rate = 1.0e-4
        weight_decay = 0.0

        # Train the student's history encoder/VAE along with its actor.
        train_vae = True

        # Train the student's depth-image encoder along with its actor.
        train_visual_encoder = True


class Go2DepthWaqDistillCfgPPO(Go2DepthWaqCfgPPO):
    """Training configuration for pure multi-teacher distillation.

    The class inherits the repository's PPO configuration layout for
    compatibility with TaskRegistry and OnPolicyRunner. PPO_WAQ_Distill ignores
    PPO-only fields such as clipping, entropy, gamma, lambda, and value loss.
    """

    runner_class_name = "DreamWaQDepthDistillRunner"

    class algorithm(Go2DepthWaqCfgPPO.algorithm):
        learning_rate = (
            Go2DepthWaqDistillCfg
            .distillation
            .learning_rate
        )

        weight_decay = (
            Go2DepthWaqDistillCfg
            .distillation
            .weight_decay
        )

        distill_target = (
            Go2DepthWaqDistillCfg
            .distillation
            .distill_target
        )

        distillation_loss_coef = (
            Go2DepthWaqDistillCfg
            .distillation
            .distillation_loss_coef
        )

        train_vae = (
            Go2DepthWaqDistillCfg.distillation.train_vae
        )

        train_visual_encoder = (
            Go2DepthWaqDistillCfg
            .distillation
            .train_visual_encoder
        )

        # These are still used by the pure-imitation optimizer.
        num_learning_epochs = 1
        num_mini_batches = 8
        max_grad_norm = 1.0

    class runner(Go2DepthWaqCfgPPO.runner):
        # The final generalist student is an ordinary depth DreamWaQ policy.
        policy_class_name = "ActorCriticDreamWaQDepth"

        # This custom class performs pure action imitation, despite retaining
        # "PPO" in its name for repository compatibility.
        algorithm_class_name = "PPO_WAQ_Distill"

        experiment_name = (
            "go2_depth_waq_multiteacher_distill"
        )
        run_name = "pure_imitation"

        max_iterations = 30000
        save_interval = 500

        # Set this only if you later want to initialize the student from an
        # existing generalist checkpoint. The LoRA teacher checkpoints are
        # configured separately above.
        pre_trained = None
        #resume = True
        #load_run = "Aug03_17-52-31_pure_imitation"
        checkpoint = -1


# export SIMULATOR=isaacgym
# export TERRAIN=baseline
# export PARKOUR_AUX=0
# unset FINETUNE DEPTHWAQ_RESUME_UNTIL

# python -m legged_gym.scripts.train \
#   --task go2_depth_waq_distill \
#   --max_iterations 30000 \
#   --headless