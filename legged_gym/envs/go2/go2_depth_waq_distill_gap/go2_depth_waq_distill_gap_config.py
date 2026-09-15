from __future__ import annotations

from legged_gym import LEGGED_GYM_ROOT_DIR

import os
from pathlib import Path

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import (
    Go2DepthWaqCfg,
    Go2DepthWaqCfgPPO,
)
from legged_gym.envs.go2.go2_depth_waq_lora.go2_depth_waq_lora_config import (
    Go2DepthWaqLoraCfg,
    Go2DepthWaqLoraCfgPPO,
)

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
        num_cols = 10
        terrain_proportions = [
            0.0,  # slope
            0.0,  # random_uniform
            0.0,  # stairs
            0.0,  # upwards_stairs
            0.0,  # discrete_obstacles
            0.0,  # stepping_stones
            1.0,  # gap
            0.0,  # pit
            0.0,  # multiple_high_platforms
            0.0,  # high_platform_gaps
            0.0,
        ]

    class rewards(Go2DepthWaqCfg.rewards):
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
        teachers = [
            {
                "name": "gap",
                "checkpoint": "/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_fft_gap/Aug12_17-01-51_dreamwaq_isaacgym/model_47000.pt",
                "teacher_actor_critic": "ActorCriticDreamWaQDepth"
            },
        ]
        terrain_type_to_teacher = [0] * 10
        distill_target = "l1"

        distillation_loss_coef = 1.0

        learning_rate = 1.0e-4
        weight_decay = 0.0
        train_vae = True
        train_visual_encoder = True


class Go2DepthWaqDistillCfgPPO(Go2DepthWaqLoraCfgPPO):
    """Training configuration for pure multi-teacher distillation.

    The class inherits the repository's PPO configuration layout for
    compatibility with TaskRegistry and OnPolicyRunner. PPO_WAQ_Distill ignores
    PPO-only fields such as clipping, entropy, gamma, lambda, and value loss.
    """

    runner_class_name = "DreamWaQDepthDistillRunner"

    class algorithm(Go2DepthWaqLoraCfgPPO.algorithm):
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

    class runner(Go2DepthWaqLoraCfgPPO.runner):
        # The final generalist student is an ordinary depth DreamWaQ policy.
        policy_class_name = "ActorCriticDreamWaQDepthLora"
        # This custom class performs pure action imitation, despite retaining
        # "PPO" in its name for repository compatibility.
        algorithm_class_name = "PPO_WAQ_Distill"
        experiment_name = (
            "go2_depth_waq_multiteacher_lora_distill_gap_try"
        )
        run_name = "pure_imitation"

        max_iterations = 10000
        save_interval = 500

        # Set this only if you later want to initialize the student from an
        # existing generalist checkpoint. The LoRA teacher checkpoints are
        # configured separately above.
        pre_trained = None
        #resume = True
        #load_run = "Aug03_17-52-31_pure_imitation"
        checkpoint = -1
