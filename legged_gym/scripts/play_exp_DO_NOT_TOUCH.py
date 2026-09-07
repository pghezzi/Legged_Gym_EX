from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *
#from legged_gym.helpers import get_load_path

import numpy as np
import torch

from legged_gym.utils.exp_data_logger import ExpLogger
from legged_gym.utils.terrain_vars import TERRAIN_INDEX, TERRAIN_KEYS
import argparse

import cv2

def get_viewed_terrain_idx(env, look_ahead_frac: float = 0.2):
    """
    Determine which terrain patch(es) the robot's depth camera is looking at.

    Projects a point forward from the base position along the current heading,
    at a fraction of the depth camera's far-plane range, then finds the
    nearest terrain origin(s) to that point (in the xy-plane).

    Args:
        env: environment instance exposing env.heading, env.simulator.base_pos,
             and env._terrain_origins.
        look_ahead_frac: fraction of the far plane (4.0) to project forward.
            0.75 -> looks ~3m ahead, a reasonable "what's dominating the FOV"
            heuristic. Use 1.0 for the far-plane edge, ~0.5 for near-field.

    Returns:
        idx: LongTensor of shape (num_envs,) — flat index into
             env._terrain_origins.reshape(-1, 3)
        row_col: LongTensor of shape (num_envs, 2) — (row, col) indices into
             the original (num_rows, num_cols, 3) grid, in case you need that
             instead of the flat index.
    """
    far_plane = 4.0
    look_ahead_dist = far_plane * look_ahead_frac

    base_pos_xy = env.simulator.base_pos[..., :2]
    base_pos = env.simulator.base_pos  # (num_envs, 3) or (3,)
    heading = env.heading              # (num_envs,)  or scalar

    # Ensure batch dim exists so this works for single-robot too
    single = base_pos.dim() == 1
    if single:
        base_pos = base_pos.unsqueeze(0)
        heading = heading.unsqueeze(0) if torch.is_tensor(heading) else torch.tensor([heading])

    device = base_pos.device
    heading = heading.to(device)

    look_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)  # (num_envs, 2)
    look_point = base_pos[:, :2] + look_dir * look_ahead_dist                  # (num_envs, 2)
    query_point = torch.where((base_pos[:, 2] < 0.25).unsqueeze(-1), base_pos_xy, look_point)

    origins = env.simulator._terrain_origins.to(device)            # (num_rows, num_cols, 3)
    num_rows, num_cols = origins.shape[0], origins.shape[1]
    origins_xy = origins[..., :2].reshape(-1, 2)         # (num_terrains, 2)

    dists = torch.cdist(query_point, origins_xy)          # (num_envs, num_terrains)
    idx = torch.argmin(dists, dim=-1)                    # (num_envs,)

    row_col = torch.stack([idx // num_cols, idx % num_cols], dim=-1)

    actual = base_pos[:, :2]

    if single:
        idx = idx.squeeze(0)
        row_col = row_col.squeeze(0)

    return idx, row_col

def _normalize_gpu_arg(gpu):
    gpu = str(gpu).strip().lower()
    if gpu.isdigit():
        return f"cuda:{gpu}"
    if gpu == "cuda":
        return gpu
    if gpu.startswith("cuda:"):
        index = gpu.split(":", 1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        f"Unsupported GPU specifier '{gpu}'. Use values like 'cuda', 'cuda:0', or '1'."
    )


def configure_runtime_device(args):
    """Normalize GPU selection and, when needed, mask visibility to the requested physical GPU.

    Genesis and some CUDA codepaths may still resolve work onto the process-local `cuda:0`.
    When a specific physical GPU is requested, we remap visibility so that local `cuda:0`
    corresponds to the requested GPU.
    """
    if getattr(args, "cpu", False):
        if hasattr(args, "gpu"):
            args.gpu = "cpu"
        if hasattr(args, "device"):
            args.device = "cpu"
        if hasattr(args, "requested_gpu"):
            args.requested_gpu = "cpu"
        return args

    requested_gpu = getattr(args, "requested_gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "device", "cuda:0")

    requested_gpu = _normalize_gpu_arg(requested_gpu)
    runtime_gpu = requested_gpu

    if requested_gpu.startswith("cuda:"):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
            local_index = int(requested_gpu.split(":", 1)[1])
            # If CUDA_VISIBLE_DEVICES is already set before Python starts, treat cuda:N
            # as a process-local device index.
            if 0 <= local_index < len(visible_gpu_ids):
                runtime_gpu = f"cuda:{local_index}"
            else:
                physical_index = str(local_index)
                if physical_index not in visible_gpu_ids:
                    raise ValueError(
                        f"Requested GPU '{requested_gpu}' is not available under "
                        f"CUDA_VISIBLE_DEVICES={visible_devices}."
                    )
                runtime_gpu = f"cuda:{visible_gpu_ids.index(physical_index)}"
        else:
            physical_index = requested_gpu.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            runtime_gpu = "cuda:0"

    elif requested_gpu == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        runtime_gpu = "cuda:0"

    args.requested_gpu = requested_gpu
    args.gpu = runtime_gpu
    if hasattr(args, "device"):
        args.device = runtime_gpu
    return args


def init_genesis(args, gs):
    """Initialize Genesis after device selection has been normalized."""
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu and args.gpu.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.gpu))

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    # Use RichHelpFormatter for colored help if available
    HAS_RICH_ARGPARSE = False
    formatter_class = RichHelpFormatter if HAS_RICH_ARGPARSE else argparse.HelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        description="LeggedGym-Ex - Train legged robots with reinforcement learning",
        epilog="For more information, visit: https://github.com/lupinjia/LeggedGym-Ex"
    )
    parser.add_argument('--gpu',            type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',         action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',           type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--motion_file',    type=str, 
                        default=None, 
                        help="motion file to load, under resources/reference_motion")
    parser.add_argument('--motion_out_dir',  type=str, default=None, help="directory to save the motion generated by the process_reference_motion script")
    parser.add_argument('--distill',         action='store_true', default=False, help="[Only used in ts_depth] whether to train a student policy with teacher-student framework")
    parser.add_argument('--teacher_model_path', type=str, default=None, help="[Only used in ts_depth] path to the teacher model for distillation, format load_run/checkpoint.pt")
    parser.add_argument('--num_student',     type=int, default=None, help="[Only used in ts_depth] number of student envs to train in parallel for distillation")

    parser.add_argument('--jit',            type=str, default='', help="path to a jit-scripted policy to load and swap with the trained policy (replaces JIT env var)")

    parser.add_argument('--test_terrain', type=str, default=None, help="current test_terrain")
    parser.add_argument('--curriculum', action='store_true', default=False, help="Load default curriculum")
    parser.add_argument('--multiterrain', action='store_true', default=False, help="multiple terrains")
    parser.add_argument('--extreme', action='store_true', default=False, help="use extreme version of test terrain")

    parser.add_argument('--save_depth_classifier_data', action='store_true', default=False, help="if with depth cam use to save depth image and rpy of all executed steps")
    parser.add_argument('--filter_depth_classifier_data', action='store_true', default=False, help="split data based on terminates")
    parser.add_argument('--no_depth_cam', action='store_true', default=False, help="disable test cam if available")

    parser.add_argument('--terrain_detector', type=str, default='', help="test a terrain detector")
    parser.add_argument('--hard_terrain_detector', action='store_true', default=False, help="using sim to determine terrain")
    parser.add_argument('--terrain_detector_jit', type=str, default='', help="test a terrain detector")
    parser.add_argument('--baysian_filter', type=str, default='', help="test a terrain detector")

    parser.add_argument('--command_test_suite', action='store_true', default=False, help="run a simple commnad test suite")
    parser.add_argument('--explore', action='store_true', default=False, help="explore terrain")

    parser.add_argument('--multitask', type=str, nargs='+', default=None, help="list of tasks to run/train on")

    parser.add_argument('--seed',           type=int, default=42, help="random seed")

    args = configure_runtime_device(parser.parse_args())

    selected = [
        name for name, enabled in (
            ("curriculum", args.curriculum),
            ("multiterrain", args.multiterrain),
            ("test_terrain", args.test_terrain is not None),
        )
        if enabled
    ]

    if len(selected) > 1:
        parser.error(
            f"Only one of --curriculum, --multiterrain, or --test_terrain may be specified. Got: {', '.join(selected)}"
        )
    
    selected = [
        name for name, enabled in (
            ("test_terrain", args.test_terrain),
            ("hard_terrain_detector", args.hard_terrain_detector),
        )
        if enabled
    ]

    if args.save_depth_classifier_data and len(selected) == 0:
        parser.error(
            f"To collect data you must have either test_terrain or hard_terrain_detector active to classify the data"
        )


    return args

def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    TERRAIN_CONFIGS = {
        "random_uniform": {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.005,
            "downsampled_scale": 0.2,
        },
        "slope": {
            "type": "terrain_utils.pyramid_sloped_terrain",
            "slope": -0.4,
            "platform_size": 3.0,
        },
        "stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width": 0.4,
            "step_height": -0.25,
            "platform_size": 3.0,
        },
        "upwards_stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width": 0.4,
            "step_height": 0.3,
            "platform_size": 3.0,
        },
        "discrete": {
            "type": "terrain_utils.discrete_obstacles_terrain",
            "max_height": 0.06,
            "min_size": 1.0,
            "max_size": 2.0,
            "num_rects": 20,
            "platform_size": 3.0,
        },
        "wave": {
            "type": "terrain_utils.wave_terrain",
            "amplitude": 0.2,
            "num_waves": 2,
        },
        "stepping_stones": {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_size": 1.0,
            "max_height": 0.1,
            "stone_distance": 0.3,
            "platform_size": 3.0,
        },
        "gap": {
            "type": "terrain_utils.gap_terrain",
            "gap_size": 0.8,
            "platform_size": 5.0,
        },
        "pit": {
            "type": "terrain_utils.pit_terrain",
            "depth": 0.52,
            "platform_size": 3.0,
        },
        "multiple_high_platforms" : {
            "type": 'terrain_utils.multiple_high_platforms_terrain',
            "high_platform_height": 0.6,
            "high_platform_length": 1,
            "high_platform_width": 1.5,
            "high_platform_interval": 1.5,
        },
        "center_platform": {
            "type": "terrain_utils.center_platform_terrain",
            "height": 0.4,
            "platform_size": 3.0,
        }
    }
    env_cfg.seed = args.seed
    if args.num_envs:
        envs = args.num_envs
    else:
        envs =  min(env_cfg.env.num_envs, 100)
    task_name = args.task
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = envs
    env_cfg.asset.terminate_after_contacts_on = []
    if args.explore:
        env_cfg.init_state.yaw_random_scale = np.pi
        env_cfg.commands.ranges.heading = [-3.14, 3.14]

    else:
        env_cfg.init_state.yaw_random_scale = 0
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = env_cfg.env.num_envs
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.commands.custom_command_curriculum = True
    env_cfg.viewer.rendered_envs_idx = list(range(min(env_cfg.env.num_envs, envs)))
    # adjust parameters according to terrain type
    #env_cfg.terrain.vertical_scale = 0.1
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        if args.curriculum:
            env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
            env_cfg.terrain.num_cols = sum(1 for i in env_cfg.terrain.terrain_proportions if i > 0)
            env_cfg.commands.custom_command_curriculum = False

            def sigfig(x, n=3):
                if x == 0:
                    return 0
                return round(x, n - int(np.floor(np.log10(abs(x)))) - 1)
            
            def flatten_dict(d, parent_key=""):
                """Flatten a nested dictionary into {key.path: value}."""
                items = {}

                for key, value in d.items():
                    new_key = f"{parent_key}.{key}" if parent_key else key

                    if isinstance(value, dict):
                        items.update(flatten_dict(value, new_key))
                    else:
                        items[new_key] = value

                return items


            def generate_curriculum_values(config, x):
                """
                Generate actual values for difficulty levels:
                    0/x, 1/x, ..., (x-1)/x
                """
                flat_config = flatten_dict(config)

                difficulties = [i / x for i in range(x)]
                result = {}

                for key, expression in flat_config.items():
                    values = []

                    for difficulty in difficulties:
                        # Numeric constants such as "0.20"
                        # and expressions such as "difficulty * 0.4"
                        # are both handled here.
                        if isinstance(expression, (int, float)):
                            value = expression
                        else:
                            value = eval(
                                expression,
                                {"np": np, "__builtins__": {}},
                                {"difficulty": difficulty},
                            )

                        values.append(sigfig(value))

                    result[key] = values

                return result

            generated = generate_curriculum_values(
                env_cfg.terrain.terrain_curriculum_difficulty,
                env_cfg.terrain.num_rows,
            )

            for key, values in generated.items():
                print(f"{key}: {values}")
            return
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.border_size = 5.0
        #if args.save_depth_classifier_data:
        #    env_cfg.init_state.yaw_random_scale = 3.14 # camera views all terrain possible
        #    env_cfg.commands.heading_command = False
        #    env_cfg.rewards.scales.tracking_lin_vel = 0.0
        #    env_cfg.commands.custom_command_curriculum = True
        #    pass
        if args.multiterrain:
            env_cfg.terrain.border_size = 0.0
            env_cfg.env.episode_length_s = 120
            env_cfg.terrain.num_rows = 10
            env_cfg.terrain.num_cols = 4
            env_cfg.terrain.platform_size = 3.0
            env_cfg.terrain.curriculum = False
            env_cfg.terrain.selected   = False
            env_cfg.terrain.custom_selected   = True
            env_cfg.custom_command_curriculum = False
            env_cfg.zero_cmd_prob = 0.0
            env_cfg.init_state.yaw_random_scale = 3.14
            #env_cfg.commands.custom_command_curriculum = True
            env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]   # min max [m/s]
            env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]   # min max [m/s]
            env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]    # min max [rad/s]
            #env_cfg.commands.ranges.heading = [0.0, 0.0]
            env_cfg.commands.ranges.heading = [-3.14, 3.14]

            import random
            rng = random.Random(args.seed)


            def make_terrain():
                terrain_type = rng.choice([
                    "random_uniform",
                    "gap",
                    "stairs_down",
                    "stairs_up",
                    "pit",
                    "central"
                ])

                if terrain_type == "random_uniform":
                    return {
                        "type": "terrain_utils.random_uniform_terrain",
                        "min_height": -0.05,
                        "max_height": 0.05,
                        "step": 0.005,
                        "downsampled_scale": 0.2,
                    }

                if terrain_type == "gap":
                    return {
                        "type": "terrain_utils.gap_terrain",
                        "gap_size": rng.uniform(0.2, 0.8),
                        "platform_size": 3.0,
                    }

                if terrain_type == "stairs_down":
                    return {
                        "type": "terrain_utils.pyramid_stairs_terrain",
                        "step_width": 0.4,
                        "step_height": rng.uniform(-0.3, -0.1),
                        "platform_size": 3.0,
                    }

                if terrain_type == "stairs_up":
                    return {
                        "type": "terrain_utils.pyramid_stairs_terrain",
                        "step_width": 0.4,
                        "step_height": rng.uniform(0.1, 0.3),
                        "platform_size": 3.0,
                    }

                if terrain_type == "pit":
                    return {
                        "type": "terrain_utils.pit_terrain",
                        "depth": rng.uniform(0.2, 0.5),
                        "platform_size": 3.0,
                    }
                if terrain_type == "central":
                    return {
                        "type": "terrain_utils.center_platform_terrain",
                        "height": rng.uniform(0.2, 0.5),
                        "platform_size": 3.0,
                    }


            env_cfg.terrain.terrain_map = [
                make_terrain()
                for _ in range(env_cfg.terrain.num_rows * env_cfg.terrain.num_cols)
            ]

        elif args.test_terrain:
            if args.save_depth_classifier_data:
                env_cfg.zero_cmd_prob = 0.0
                env_cfg.init_state.yaw_random_scale = 3.14
                #env_cfg.commands.custom_command_curriculum = True
                env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]   # min max [m/s]
                env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]   # min max [m/s]
                env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]    # min max [rad/s]
                #env_cfg.commands.ranges.heading = [0.0, 0.0]
                env_cfg.commands.ranges.heading = [-3.14, 3.14]
            if args.test_terrain == "plane":
                if env_cfg.terrain.mesh_type == "heightfield":
                    env_cfg.terrain.curriculum = True
                    env_cfg.terrain.selected   = False
                    env_cfg.terrain.terrain_proportions = [0] * 11
                    env_cfg.terrain.terrain_kwargs = {}
                elif env_cfg.terrain.mesh_type == "trimesh":
                    env_cfg.terrain.curriculum = False
                    env_cfg.terrain.selected   = True
                    env_cfg.terrain.terrain_kwargs = {
                        "type": "terrain_utils.random_uniform_terrain",
                        "min_height": 0,
                        "max_height": 0,
                        "step": 0.005,
                        "downsampled_scale": 0.2,
                    }
            else:
                env_cfg.terrain.curriculum = False
                env_cfg.terrain.selected   = True
                if args.extreme:
                    if args.test_terrain == "baseline":
                        env_cfg.terrain.terrain_kwargs = {
                            "type": "terrain_utils.random_uniform_terrain",
                            "min_height": -0.05,
                            "max_height": 0.05,
                            "step": 0.005,
                            "downsampled_scale": 0.2,
                        }
                    elif args.test_terrain == "gap":
                        from types import SimpleNamespace
                        env_cfg.termination = SimpleNamespace(
                            reset_unrecoverable_gaps=True,
                            gap_terrain_depth_threshold=1.0,
                            gap_foot_drop_threshold=0.25,
                            gap_base_drop_threshold=0.30,
                            gap_min_fallen_feet=1,
                            gap_reset_steps=4,
                        )
                        env_cfg.terrain.terrain_kwargs = {
                            "type": "terrain_utils.gap_terrain",
                            "gap_size": 0.8,
                            "platform_size": 3.0,
                        }
                    elif args.test_terrain == "stairs":
                        env_cfg.terrain.terrain_kwargs = {
                            "type": "terrain_utils.pyramid_stairs_terrain",
                            "step_width": 0.4,
                            "step_height": -0.3,
                            "platform_size": 3.0,
                        }
                else:
                    if args.test_terrain == "baseline":
                        env_cfg.terrain.terrain_kwargs = TERRAIN_CONFIGS["random_uniform"]
                    elif args.test_terrain == "all_stairs":
                        env_cfg.terrain.curriculum = True
                        env_cfg.terrain.selected   = False
                        env_cfg.terrain.num_rows = 10
                        env_cfg.terrain.num_cols = 2
                        env_cfg.terrain.terrain_curriculum_difficulty["step_height"] = "0.2"
                        env_cfg.terrain.terrain_proportions = [0] * 11
                        env_cfg.terrain.terrain_proportions[TERRAIN_INDEX["stairs"]] = 0.5
                        env_cfg.terrain.terrain_proportions[TERRAIN_INDEX["upwards_stairs"]] = 0.5
                    else:
                        env_cfg.terrain.terrain_kwargs = TERRAIN_CONFIGS[args.test_terrain]
                    print(env_cfg.terrain.terrain_kwargs)

        #env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
        #                                  "min_height" : -0.05, "max_height": 0.05, 
        #                                  "step":0.005, "downsampled_scale" : 0.2}

        #env_cfg.terrain.terrain_kwargs = []

        # random uniform terrain
        #env_cfg.terrain.terrain_kwargs.append({"type": "terrain_utils.random_uniform_terrain", 
        #                                  "min_height" : -0.05, "max_height": 0.05, 
        #                                  "step":0.005, "downsampled_scale" : 0.2})
        # # slope
        #env_cfg.terrain.terrain_kwargs.append({"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0})
        # stairs
        #env_cfg.terrain.terrain_kwargs.append({"type": "terrain_utils.pyramid_stairs_terrain",
        #                                 "step_width": 0.31, "step_height": 0.1, "platform_size": 3.0})

        #env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.06,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}

        #env_cfg.terrain.num_sub_terrains = len(env_cfg.terrain.terrain_kwargs)
        
        # # discrete obstacles
        #env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.06,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        #env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.2, "num_waves": 2}
        # stepping stones
        #env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_size": 1.0, "max_height": 0.1,
        #                                   "stone_distance": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.gap_terrain", 
        #                                   "gap_size": 0.2, "platform_size": 3.0}
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pit_terrain", 
        #                                   "depth": 0.2, "platform_size": 3.0}
    else:
        for i in range(2):
            env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
            env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4    
        
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    if args.use_joystick or args.command_test_suite:
        env_cfg.commands.heading_command = False
        env_cfg.rewards.scales.tracking_lin_vel = 0.0
        env_cfg.commands.custom_command_curriculum = True
    
    # env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    # env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]

    #env_cfg.commands.ranges.lin_vel_x   = [-1.0, 1.0]
    #env_cfg.commands.ranges.lin_vel_y   = [-1.0, 1.0]
    #env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]

    #env_cfg.commands.ranges.heading = [0.0, 0.0]
    
    #env_cfg.init_state.yaw_random_scale = 0.2

    # Turn off/on domain randomization elements
    env_cfg.noise.add_noise = True
    # Disable some of the domain randomization (our payload will handle that now)
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False           # Maybe keep this on?
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True

    env_cfg.asset.fix_base_link = False
    # env_cfg.env.debug_viz = False
    # env_cfg.env.debug = False
    # env_cfg.env.debug_draw_terrain_height_points = False

    #if args.record_frames or args.follow_robot:
    #    print("Adding Camera!")
    #    env_cfg.viewer.add_camera = True  # use a extra camera for moving

    

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"ankle pitch: {env.simulator.dof_pos[robot_index, [3,7]].cpu().numpy()}")
    pass

def interaction_loop(train_cfg, env, policy, args, new="", policy1=None):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
    """
    
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    env.reset()

    # logger = ExpLogger(train_cfg.runner.exp_data_path)

   

    if args.terrain_detector:
        from legged_gym.scripts.evaluation.high_level_evaluation import create_classifier, auto_load_checkpoint_bayes_filter, RuntimeClassifier
        import copy

        def make_terrain_detector(ckpt_dir):
            classifier = create_classifier(ckpt_dir, env.device)
            bayes_template = auto_load_checkpoint_bayes_filter(ckpt_dir, env.device)
            filters = [copy.deepcopy(bayes_template) for _ in range(env.num_envs)]
            return classifier, filters
        terrain_detector, bayes_filters = make_terrain_detector(args.terrain_detector)

    if "lora" in train_cfg.runner.experiment_name.lower():
        folder = "go2_lora_seq_terrain_tests"
    else:
        folder = "go2_fft_seq_terrain_tests"
    if new:
        exp_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'exp_logs', folder, f"{train_cfg.runner.experiment_name}_{new}.csv")
    else:
        exp_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'exp_logs', folder, f"{train_cfg.runner.experiment_name}.csv")

    logger = ExpLogger(exp_log_root, ref_key='q_actual', length_limit=100)

    if args.save_depth_classifier_data:
        depth_images_log = []
        base_rpy_log = []
        base_ang_vel_log = []
        resets_log = []
        terrain_name_log = []
    
    # Get initial observations according to task type
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:  # teacher-student
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif "ee" in task_name:  # explicit estimator
        estimator_features, _, _ = env.get_observations()
    elif "depth_waq" in task_name:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, depth = env.get_observations()
    elif "waq" in task_name:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    elif "pact" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
    else: # vanilla
        obs = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        from legged_gym.scripts.joystick import Joystick
        joystick = Joystick(joystick_type=args.joystick_type)
    if args.jit:
        import threading
        requested_mode = None
        lock = threading.Lock()
        import copy
        #policy_tester = torch.jit.load(args.jit,  'cpu')
        #policy_tester.swap(policy_tester.num_of_loras - 1)
        #policy_tester.swap(-1)
        #def keyboard_thread():
        #    nonlocal requested_mode
        #    while True:
        #        key = input("> ")
        #        if key:
        #            key = key.strip()[-1]
        #        else:
        #            continue
        #        with lock:
        #            if key.isnumeric():
        #                requested_mode = int(key)
        #            elif key == "-":
        #                requested_mode = -1

        t = args.test_terrain
        if t is not None:
            if t == "random_uniform":
                policy.swap(-1)
            if "stairs" in t:
                policy.swap(1)
            if t == "gap":
                policy.swap(0)
            if t in ("pit", "center_platform"):
                policy.swap(2)

        #threading.Thread(target=keyboard_thread, daemon=True).start()

    
    # env.commands[:, 0] = 0.5
    # env.commands[:, 1] = 0
    # env.commands[:, 2] = 0
    # env.commands[:, 3] = 0
    
    #if args.record_frames:
    #    env.simulator._floating_camera.start_recording()
    # interaction loop
    #env_ids = torch.arange(env.num_envs)
    #if env.simulator.custom_origins:
    #    base_pos = env.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
    #    base_pos += env.simulator.env_origins[env_ids]
    #    #base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=env.device) # xy position within 1m of the center
    #else:
    #    base_pos = env.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
    #    base_pos += env.simulator.env_origins[env_ids]
    #base_quat = quat_from_euler_xyz(
    #    torch.full((len(env_ids), 1),       0, device=env.device), 
    #    torch.full((len(env_ids), 1),       0, device=env.device), 
    #    torch.full((len(env_ids), 1), np.pi/2, device=env.device)
    #)
    #base_lin_vel = torch_rand_float(0.5, 1.0, (len(env_ids), 3), env.device)
    #base_ang_vel = torch_rand_float(0, 0, (len(env_ids), 3), env.device)
    #print(
    #    base_pos.device,
    #    base_quat.device,
    #    base_lin_vel.device,
    #    base_ang_vel.device
    #)
    #env.simulator.reset_root_states(
    #    env_ids,
    #    base_pos,
    #    base_quat,
    #    base_lin_vel,
    #    base_ang_vel
    #)
    #env._resample_commands(torch.arange(env.num_envs))
    cho = torch.tensor([0, 3.14/2, 3.14, -3.14/2, -3.14])
    commands = torch.zeros_like(env.commands)
    commands[:, 0] = 0
    commands[:, 1] = 0
    commands[:, 2] = 0
    commands[:, 3] = cho[torch.randint(0, cho.shape[0], (env.num_envs,))]
    if args.save_depth_classifier_data:
        print(f"Num of samples generated: {(10* 1000 * env.num_envs) / 5} (aprox)")
    #(10* 1000 * 100) / 5 = 2000
    for i in range(int(10.00*env.max_episode_length)):
        if not args.headless and args.follow_robot:
            pos = env.simulator.base_pos[0].cpu().numpy() + np.array(env.cfg.viewer.pos)
            lookat = env.simulator.base_pos[0].cpu().numpy() + np.array(env.cfg.viewer.lookat)
            env.set_viewer_camera(pos, lookat)
        if args.command_test_suite:
            current = i // env.max_episode_length
            factor = (((i % env.max_episode_length) // (env.max_episode_length // 2)) * (-2) + 1)
            if current == 0:
                env.commands[:, 0] = 1 * factor
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
            elif current == 1:
                env.commands[:, 0] = 0
                env.commands[:, 1] = 1 * factor
                env.commands[:, 2] = 0
            elif current == 2:
                env.commands[:, 0] = 0
                env.commands[:, 1] = 0
                env.commands[:, 2] = 1 * factor
            elif current == 3:
                env.commands[:, 0] = 0
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
        elif args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        elif i % env.max_episode_length == 0:
            env._resample_commands(torch.arange(env.num_envs))

        if args.multiterrain:
            env.commands[:, :] = commands
            if i % env.max_episode_length == 0:
                commands[:, 0] = 1
                commands[:, 1] = 0
                commands[:, 2] = 0
                commands[:, 3] = env.heading

        #print(env.commands)
        #print(env.commands)
        #if args.multiterrain:
        #    dx = env.simulator.base_pos[:, 0] - env.simulator.base_pos[:, 0]
        #    dy = env.simulator.env_origins[:, 1] - env.simulator.base_pos[:, 1]
        #    k = 0.5
        #    desired_heading = k*torch.atan2(dy, dx)
        #    env.commands[:, 3] = desired_heading
        if args.jit:
            import time
            with lock:
                if requested_mode is not None:
                    if requested_mode < policy.num_of_loras:
                        policy.swap(requested_mode)
                        start = time.perf_counter()
                        policy_tester.swap(requested_mode)
                        end = time.perf_counter()
                        print(f"Time: {end - start} seconds")
                        requested_mode = None


        #if arg.test_terrain not in ("plane", "baseline") and args.save_depth_classifier_data:
        #    env.commands[:, 2] = 0
            #print("swap")
            #policy.swap(-1)
        #if args.jit and i == 2*env.max_episode_length:
        #    print("swap")
        #    policy.swap(SWAP)
        #if i % env.max_episode_length == 0:
        #    env.commands[:, 0] = torch.empty(env.num_envs, device=env.device).uniform_(-1.0, 1.0)
        #    env.commands[:, 1] = torch.empty(env.num_envs, device=env.device).uniform_(-1.0, 1.0)
        #    env.commands[:, 2] = torch.empty(env.num_envs, device=env.device).uniform_(-1.0, 1.0)
        #    env.commands[:, 3] = torch.empty(env.num_envs, device=env.device).uniform_(-3.14, 3.14)
        
        # update commands from joystick
        
        
        # set the viewer camera to follow the first environment by default
        # TODO - fix recording/general camera follow conflict
        #if args.follow_robot:
        #    pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
        #    lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
        #    # env.set_viewer_camera(pos, lookat)
        #    env.set_camera(pos, lookat)
        #    env.simulator._floating_camera.render()
        
        # Step the environment according to task type
        if "ts" in task_name or "cat" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
            
        elif "ee" in task_name:
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif "depth_waq" in task_name:
            #print(obs_buf.shape, obs_history.shape, depth.shape)
            actions = policy(obs_buf, obs_history, depth)
            #if policy1 is not None:
            #    print(f"{torch.sum(torch.abs(actions - policy1(obs_buf, obs_history, depth)))}")
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos, depth = env.step(actions.detach())
            if not args.no_depth_cam:
                cv2.imshow("Depth", ((env.depth_sensor_output[0].squeeze())*255).to(torch.uint8).cpu().numpy())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if i % 5 == 0: # this assumes a 50hz policy so that we actually run at 10. If change, change this
                if args.save_depth_classifier_data:
                    depth_images_log.append(env.depth_sensor_output.detach().cpu().clone())
                    base_rpy_log.append(env.simulator._base_euler.detach().cpu().clone())
                    base_ang_vel_log.append(env.simulator.base_ang_vel.detach().cpu().clone())
                if args.terrain_detector:
                    _depth = env.depth_sensor_output
                    _euler = env.simulator._base_euler.detach()
                    _angve = env.simulator.base_ang_vel.detach()
                    probabilities = terrain_detector.predict(_depth, _euler, _angve)
                    for idx in range(env.num_envs):
                        probability = probabilities[idx]
                        bayes_step = bayes_filters[idx].update(probability)
                        if idx == 0:
                            print(bayes_step.label)
                if args.hard_terrain_detector:
                    look_ahead_frac = 0
                    _, row_col = get_viewed_terrain_idx(env, look_ahead_frac=look_ahead_frac)
                    row_col = row_col.cpu()
                    labels = np.atleast_1d(
                        env.simulator._terrain.labels[
                            row_col[:, 0],
                            row_col[:, 1]
                        ]
                    )
                    if args.save_depth_classifier_data:
                        terrain_name_log.append(labels)
                    if args.jit or args.multitask:
                        policy.set_labels(labels)
        elif "waq" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())            
        elif "pact" in task_name:
            # print("obs_buf - ", obs_buf.cpu().numpy())
            # print("obs_history - ", obs_history.cpu().numpy())
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())


        #terminate at bound:
        if not args.multiterrain and args.test_terrain != "plane":
            x_out_of_bound = (env.simulator.base_pos[:, 0] < 0.0) | (env.simulator.base_pos[:, 0] > env.cfg.terrain.num_rows * env.cfg.terrain.terrain_length)
            y_out_of_bound = (env.simulator.base_pos[:, 1] < 0.0) | (env.simulator.base_pos[:, 1] > env.cfg.terrain.num_cols * env.cfg.terrain.terrain_width)
            out = x_out_of_bound | y_out_of_bound
            dones |= out
            env_ids = out.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                env.simulator._terrain_levels[env_ids] = env.simulator._max_terrain_level
                env.reset_idx(env_ids)

        #keep this?????
        if args.save_depth_classifier_data and not args.multiterrain and (env.cfg.terrain.terrain_proportions[TERRAIN_INDEX["gap"]] > 0 or args.test_terrain == "gap"):
            difficulty = env.simulator.terrain_levels / env.cfg.terrain.num_rows
            dist = torch.norm(env.simulator.base_pos - env.simulator.env_origins, dim=-1)
            bound = env.cfg.terrain.platform_size/2 + eval(env.cfg.terrain.terrain_curriculum_difficulty["gap_size"]) + (-0.1 if args.filter_depth_classifier_data else 1)
            out = dist >= bound
            dones |= out
            env_ids = out.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                env.simulator._terrain_levels[env_ids] = env.simulator._max_terrain_level
                env.reset_idx(env_ids)

        if i % 5 == 0 and args.save_depth_classifier_data and args.filter_depth_classifier_data:
            resets_log.append(dones)

        #if dones[0] == True:
        #    if args.terrain_detector:
        #        terrain_detector.reset_temporal_filter()
        #    if args.baysian_filter:
        #        terrain_detector.reset()
        
        #print(env.commands[0, :])
        # print debug info
        print_debug_info(env, robot_index)
        
        # # Update logger info
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
        #             'command_x': env.commands[robot_index, 0].item(),
        #             'command_y': env.commands[robot_index, 1].item(),
        #             'command_yaw': env.commands[robot_index, 2].item(),
        #             'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.simulator.link_contact_forces[robot_index, 
        #                                                                   env.simulator.feet_indices, 2].cpu().numpy()
        #         }
        #     )
        # elif i==stop_state_log:
        #     logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

        #logger.log_states(
        #    {
        #        'base_cmd':env.commands.detach().cpu().numpy().tolist(),
        #        'base_pose':env.simulator.base_pos.detach().cpu().numpy().tolist(),
        #        'base_rpy':env.simulator.base_euler.detach().cpu().numpy().tolist(),
        #        'q_actual':env.simulator.dof_pos.detach().cpu().numpy().tolist(),
        #        'base_lin_vel':env.simulator.base_lin_vel.detach().cpu().numpy().tolist(),
        #        'base_ang_vel':env.simulator.base_ang_vel.detach().cpu().numpy().tolist(),
        #        'dof_vel':env.simulator.dof_vel.detach().cpu().numpy().tolist(),
        #        'proj_grav':env.simulator.projected_gravity.detach().cpu().numpy().tolist(),
        #        'feet_pos':env.simulator.feet_pos.detach().cpu().numpy().tolist(),
        #        'failure':list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist())),
        #    })
    if "depth_waq" in task_name and not args.no_depth_cam:
        cv2.destroyAllWindows()
    
    def reset_split(tensor, reset_tensor, outlier_thresh=0.2):
        eps = tensor.shape[0]
        envs = tensor.shape[1]

        data = []
        for env in range(envs):
            splits = torch.nonzero(reset_tensor[:, env]).squeeze(1)
            last_split = 0
            for split in splits:
                t = tensor[last_split:split, env]
                data.append(t)
                last_split = split
            data.append(tensor[last_split:, env])
        lengths = torch.tensor([t.shape[0] for t in data], dtype=torch.float)
        median = lengths.median()
        keep = (lengths >= median * (1 - outlier_thresh))
        kept = [t for t, k in zip(data, keep) if k]
        target_len = min(t.shape[0] for t in kept)
        final = [t[:target_len] for t in kept]
        return torch.stack(final, dim=1)
    
    def terrain_keys(label):
        if "stairs" in label:
            return "stairs"
        elif "pit" in label or "center_platform" in label:
            return "pit"
        return label

    def tensor_to_keys(t):
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                return terrain_keys(TERRAIN_KEYS[t.item()])
            return [tensor_to_keys(x) for x in t] 
        if isinstance(t, list):
            return [tensor_to_keys(x) for x in t]
        return terrain_keys(TERRAIN_KEYS[t])
    
    def get_shape(x):
        shape = []
        while isinstance(x, list):
            shape.append(len(x))
            x = x[0] if x else []
        return tuple(shape)
    
    if args.save_depth_classifier_data:
        print("compiling")
        depth_images_tensor = torch.stack(depth_images_log, dim=0).squeeze(2)
        del depth_images_log
        base_rpy_tensor = torch.stack(base_rpy_log, dim=0)
        del base_rpy_log
        base_ang_vel_tensor = torch.stack(base_ang_vel_log, dim=0)
        del base_ang_vel_log
        if terrain_name_log:
            labels_tensor = torch.from_numpy(np.stack(terrain_name_log)).detach().cpu()
            del terrain_name_log
        else:
            labels_tensor = None
        print("splitting")
        if args.filter_depth_classifier_data:
            reset_tensor = torch.stack(resets_log, dim=0)
            print(depth_images_tensor.shape)
            depth_images_tensor = reset_split(depth_images_tensor, reset_tensor)   # [T, num_envs, H, W] (or however depth is shaped)
            print(depth_images_tensor.shape)
            base_rpy_tensor = reset_split(base_rpy_tensor, reset_tensor)       # [T, num_envs, 4]
            base_ang_vel_tensor = reset_split(base_ang_vel_tensor, reset_tensor)
            if labels_tensor is not None:
                labels_tensor = reset_split(labels_tensor, reset_tensor)
        if labels_tensor is not None:
            labels_tensor = tensor_to_keys(labels_tensor)
        if args.multitask:
            save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'depth_waq_selector', 'depth_data', 'multitask')
        else:
            save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'depth_waq_selector', 'depth_data', train_cfg.runner.experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_name = (
            f"{train_cfg.runner.experiment_name}"
            f"{'_filtered' if args.filter_depth_classifier_data else ''}"
            f"{'_test_terrain_' + args.test_terrain + '_' if args.test_terrain else ''}"
            f"{'_curriculum' if args.curriculum else ''}"
            f"{'_explore' if args.explore else ''}"
            f"{'_extreme' if args.extreme else ''}"
            f"_seed_{args.seed}"
            f"_{env.cfg.env.num_envs}_capture_{timestamp}.pt"
        )

        save_path = os.path.join(save_dir, save_name)
        if args.test_terrain:
            if "stairs" in args.test_terrain:
                test_terrain = "stairs"
            elif "pit" in args.test_terrain:
                test_terrain = "pit"
            else:
                test_terrain = args.test_terrain

        labels_list = labels_tensor if labels_tensor is not None else [
                [test_terrain] * depth_images_tensor.shape[1]
                for _ in range(depth_images_tensor.shape[0])
            ]

        print("Saving file")

        torch.save({
            "depth_images": depth_images_tensor,
            "depth_images_shape": tuple(depth_images_tensor.shape),
            "base_rpy": base_rpy_tensor,
            "base_rpy_shape": tuple(base_rpy_tensor.shape),
            "base_ang_vel": base_ang_vel_tensor,
            "base_ang_vel_shape": tuple(base_ang_vel_tensor.shape),
            "terrain_name": labels_list,
            "filtered": args.filter_depth_classifier_data
        }, save_path)

        print(f"Saved depth images {tuple(depth_images_tensor.shape)}, "
          f"base rpy {tuple(base_rpy_tensor.shape)}, "
          f"base ang vel {tuple(base_ang_vel_tensor.shape)}, "
          f"terrain name list {get_shape(labels_list)}"
          f"and terrain name to: {save_path}")
        print(save_path)

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:\        save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'depth_waq_selector', 'depth_data', train_cfg.runner.experiment_name)

        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "depth_waq" in task_name:
        exporter = PolicyExporterDepthWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, train_cfg)
    elif "pact" in task_name:
        exporter = PolicyExporterPACT(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    
class multi_jit:
    def __init__(self, policy, terrain_keys):
        """
        policy:
            Your existing JIT policy containing the different policies.

        terrain_keys:
            TERRAIN_KEYS, e.g.
            {
                0: "gap",
                1: "stairs",
                2: "pit",
                ...
            }
        """
        self.policy = policy
        self.terrain_keys = terrain_keys

        # Map terrain name -> policy index
        self.terrain_to_policy = {
            "random_uniform": -1,
            "gap": 0,
            "stairs": 1,
            "pit": 2,
            "center_platform": 2,
        }

    def _get_policy_index(self, label):
        terrain = self.terrain_keys[label]
        if terrain == "random_uniform":
            return -1
        if terrain == "gap":
            return 0
        if "stairs" in terrain:
            return 1
        if terrain in ("pit", "center_platform"):
            return 2

        raise ValueError(f"Unknown terrain: {terrain}")

    def __call__(self, obs_buf, obs_history, depth):
        """
        Run the appropriate policy for each environment.

        Returns:
            Same output shape as policy(obs_buf, obs_history, depth)
        """
        num_envs = obs_buf.shape[0]

        # Get terrain labels from wherever you store them.
        # This assumes they have already been provided to this class.
        labels = self.labels

        # First environment determines output shape/device/dtype.
        output = None

        # Group environments by policy
        policy_envs = {}

        for env_id in range(num_envs):
            #print("HERE", env_id, labels)
            x = labels[env_id]
            policy_idx = self._get_policy_index(labels[env_id])

            if policy_idx not in policy_envs:
                policy_envs[policy_idx] = []

            policy_envs[policy_idx].append(env_id)

        # Run each policy only on the environments that need it
        for policy_idx, env_ids in policy_envs.items():
            env_ids = torch.tensor(
                env_ids,
                device=obs_buf.device,
                dtype=torch.long,
            )

            # Select batch
            obs = obs_buf[env_ids]
            history = obs_history[env_ids]
            d = depth[env_ids]

            # Select policy
            self.policy.swap(policy_idx)

            actions = self.policy(obs, history, d)

            # Allocate output after seeing policy output
            if output is None:
                output = torch.empty(
                    (num_envs,) + actions.shape[1:],
                    device=actions.device,
                    dtype=actions.dtype,
                )

            output[env_ids] = actions

        return output

    def set_labels(self, labels):
        self.labels = labels
    
    def swap(self, index):
        pass


class normal_jit:
    def __init__(self, policy):
        """
        policy:
            Your existing JIT policy containing the different policies.

        terrain_keys:
            TERRAIN_KEYS, e.g.
            {
                0: "gap",
                1: "stairs",
                2: "pit",
                ...
            }
        """
        self.policy = policy


    def __call__(self, obs_buf, obs_history, depth):
        return self.policy(obs_buf, obs_history, depth)

    def set_labels(self, labels):
        pass
    
    def swap(self, index):
        pass

class multi_policy:
    def __init__(self, policies, terrain_keys):
        """
        policies:
            List of inference policies, e.g.
                runner = runner_class(env, train_cfg_dict, "/", env.device)
                runner.load(ckpt)
                policies.append(runner.get_inference_policy(device=env.device))
            Indexed the same way _get_policy_index() returns indices.

        terrain_keys:
            TERRAIN_KEYS, e.g.
            {
                0: "gap",
                1: "stairs",
                2: "pit",
                ...
            }
        """
        self.policies = policies
        self.terrain_keys = terrain_keys
        self.labels = None

    def _get_policy_index(self, label):
        terrain = self.terrain_keys[label]

        if terrain == "random_uniform":
            return 0
        if terrain == "gap":
            return 1
        if "stairs" in terrain:
            return 2
        if terrain in ("pit", "center_platform"):
            return 3

        raise ValueError(f"Unknown terrain: {terrain}")

    def __call__(self, obs_buf, obs_history, depth):
        """
        Run the appropriate policy for each environment.

        Returns:
            Same output shape as policy(obs_buf, obs_history, depth)
        """
        num_envs = obs_buf.shape[0]

        # Get terrain labels from wherever you store them.
        labels = self.labels

        output = None

        # Group environments by policy index
        policy_envs = {}
        for env_id in range(num_envs):
            policy_idx = self._get_policy_index(labels[env_id])
            policy_envs.setdefault(policy_idx, []).append(env_id)

        # Run each policy only on the environments that need it
        for policy_idx, env_ids in policy_envs.items():
            env_ids = torch.tensor(
                env_ids,
                device=obs_buf.device,
                dtype=torch.long,
            )

            # Select batch
            obs = obs_buf[env_ids]
            history = obs_history[env_ids]
            d = depth[env_ids]

            # Select policy — direct list indexing instead of .swap()
            policy = self.policies[policy_idx]
            actions = policy(obs, history, d)

            # Allocate output after seeing first policy's output
            if output is None:
                output = torch.empty(
                    (num_envs,) + actions.shape[1:],
                    device=actions.device,
                    dtype=actions.dtype,
                )

            output[env_ids] = actions

        return output

    def set_labels(self, labels):
        self.labels = labels
    
    def swap(self, index):
        pass

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    #randomly divides robots
    #env.simulator._terrain_levels[:] = env.simulator._max_terrain_level + 1
    #env.reset_idx(torch.arange(env.num_envs))
    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)
    policy1 = None
    
    if args.jit:
        policy1 = policy
        policy = torch.jit.load(args.jit,  map_location=args.gpu if not args.cpu else 'cpu')
        if hasattr(policy, 'swap'):
            policy = multi_jit(policy, TERRAIN_KEYS)
        else:
            policy = normal_jit(policy)
        policy.set_labels(torch.tensor([1]*env.num_envs))
    elif args.multitask:
        from rsl_rl.utils.runner_registry import runner_registry
        runner_class = runner_registry.get_runner_class(train_cfg.runner_class_name)
        train_cfg_dict = class_to_dict(train_cfg)
        policies = []
        for ckpt in args.multitask:
            runner = runner_class(env, train_cfg_dict, "/", env.device)
            runner.load(ckpt)
            policies.append(
                runner.get_inference_policy(device=env.device)
            )
        policy = multi_policy(policies, TERRAIN_KEYS)
        policy.set_labels(torch.tensor([1]*env.num_envs))  
    else:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        path = os.path.join(os.path.dirname(path), 'current_actor_args.pt')
        temp = class_to_dict(train_cfg.policy)
        temp.update({
            "num_actor_obs": env.num_obs,
            "num_actions": env.num_actions,
            "num_privileged_obs": env.num_privileged_obs,
            "num_history_input": env.num_history_obs,
            "num_latent_dims": env.num_latent_dims,
            "num_explicit_dims": env.num_explicit_dims,
            "num_decoder_output": env.num_decoder_output,
        })
        torch.save({
            "args": temp
        },
        path)

    # export policy as a jit module (used to run it from C++ or python)
    #if train_cfg.runner.load_run == -1:
    #    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    #    try:
    #        runs = os.listdir(log_root)
    #        #TODO sort by date to handle change of month
    #        runs.sort()
    #        if 'exported' in runs: runs.remove('exported')
    #        train_cfg.runner.load_run = runs[-1]
    #    except:
    #        raise ValueError("No runs in this directory: " + root)
    #path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
    #                        train_cfg.runner.load_run, 'exported')
    # export_policy(ppo_runner, path, args, env_cfg, train_cfg)
    
    interaction_loop(train_cfg, env, policy, args, args.test_terrain, policy1 if policy1 else None)

    #if args.record_frames:
    #    try:
    #        filename_mp4 = f"{train_cfg.runner.experiment_name}_discrete_normal_viz.mp4"
    #    except:
    #        from datetime import datetime
    #        filename_mp4 = f"{datetime.now().timestamp()}"
    #    
    #    env.simulator._floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
    #    print("Saved recording to " + filename_mp4)
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
