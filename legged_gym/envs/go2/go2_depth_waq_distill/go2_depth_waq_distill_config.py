from __future__ import annotations

from legged_gym import LEGGED_GYM_ROOT_DIR

import os
from pathlib import Path

from dotenv import load_dotenv

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import (
    Go2DepthWaqCfg,
    Go2DepthWaqCfgPPO,
)

# This file is located at:
#
# Legged_Gym_EX/
# └── legged_gym/
#     └── envs/
#         └── go2/
#             └── go2_depth_waq_distill/
#                 └── go2_depth_waq_distill_config.py
#
# parents[4] therefore points to the Legged_Gym_EX repository root.

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
    """Environment configuration for multi-teacher LoRA distillation."""

    class viewer(Go2DepthWaqCfg.viewer):
        # Rendering only the two camera environments keeps visual smoke tests
        # lightweight. CLI runs with one environment clamp this list to [0].
        rendered_envs_idx = [0, 1]

    class terrain(Go2DepthWaqCfg.terrain):
        # Keep the ten curriculum columns used by Go2DepthWaqCfg, but dedicate
        # the first half to stairs and the second half to gaps. The indices
        # here follow TERRAIN_KEYS in legged_gym.utils.terrain_vars.
        num_cols = 12
        terrain_proportions = [
            0.0,  # slope
            0.0,  # random_uniform
            0.25,  # stairs
            0.25,  # upwards_stairs
            0.0,  # discrete_obstacles
            0.0,  # stepping_stones
            0.25,  # gap
            0.25,  # pit
            0.0,  # multiple_high_platforms
            0.0,  # high_platform_gaps
            0.0,
        ]

    class rewards(Go2DepthWaqCfg.rewards):
        # Gap and stairs teachers were trained with the same reward scales.
        # Define them explicitly because the parent config selects its scales
        # at import time from TERRAIN, whose default is random_uniform.
        class scales:
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
        # All teachers use this baseline unless a teacher dictionary supplies
        # its own "base_model" override.
        base_model = required_env("DISTILL_BASE_MODEL")

        # All LoRA components use this rank unless a teacher dictionary
        # overrides "rank" or a component-specific rank.
        lora_rank = optional_int_env(
            "DISTILL_LORA_RANK",
            default=8,
        )

        # ------------------------------------------------------------------
        # LoRA teachers
        # ------------------------------------------------------------------
        #
        # List order defines the numeric teacher ID:
        #
        #   teachers[0] -> teacher ID 0 -> gap
        #   teachers[1] -> teacher ID 1 -> stairs
        #
        # The checkpoint paths are stored in the local .env file rather than
        # directly in this committed configuration file.
        teachers = [
            {
                "name": "gap",
                "checkpoint": "/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_gap/Aug12_17-01-51_dreamwaq_isaacgym/model_47000.pt",
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
            {
                "name": "stairs",
                "checkpoint": "/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_all_stairs/Aug14_14-23-53_dreamwaq_isaacgym/model_47000.pt",
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
            {
                "name": "pit",
                "checkpoint": "/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_pit/Aug29_00-30-31_dreamwaq_isaacgym/model_67000.pt",
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
        ]

        # ------------------------------------------------------------------
        # Terrain column -> teacher mapping
        # ------------------------------------------------------------------
        #
        # Index:
        #   Genesis terrain type/column ID
        #
        # Value:
        #   index into the teachers list above
        #
        # Curriculum generation assigns the lower-numbered columns to stairs
        # and the higher-numbered columns to gaps. Teacher 1 is stairs and
        # teacher 0 is gap, matching the teachers list above.
        terrain_type_to_teacher = [1] * 6 + [0] * 3 + [2] * 3

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

        max_iterations = 20000
        save_interval = 500

        # Set this only if you later want to initialize the student from an
        # existing generalist checkpoint. The LoRA teacher checkpoints are
        # configured separately above.
        pre_trained = None
        #resume = True
        #load_run = "Aug03_17-52-31_pure_imitation"
        checkpoint = -1