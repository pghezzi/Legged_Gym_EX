from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *
#from legged_gym.helpers import get_load_path

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.exp_data_logger import ExpLogger
from legged_gym.utils.terrain_vars import TERRAIN_INDEX
import argparse


SWAP = 1

command_tests = {
    "forward": {
        "vx": 1.0, "vy": 0.0, "wz": 0.0, "heading": 0.0,
    },
    "backward": {
        "vx": -1.0, "vy": 0.0, "wz": 0.0, "heading": 0.0,
    },
    "left_strafe": {
        "vx": 0.0, "vy": 1.0, "wz": 0.0, "heading": 0.0,
    },
    "right_strafe": {
        "vx": 0.0, "vy": -1.0, "wz": 0.0, "heading": 0.0,
    },
    "forward_left": {
        "vx": 1.0, "vy": 1.0, "wz": 0.0, "heading": 0.0,
    },
    "forward_right": {
        "vx": 1.0, "vy": -1.0, "wz": 0.0, "heading": 0.0,
    },
    "backward_left": {
        "vx": -1.0, "vy": 1.0, "wz": 0.0, "heading": 0.0,
    },
    "backward_right": {
        "vx": -1.0, "vy": -1.0, "wz": 0.0, "heading": 0.0,
    },
    "rotate_ccw": {
        "vx": 0.0, "vy": 0.0, "wz": 1.0, "heading": 0.0,
    },
    "rotate_cw": {
        "vx": 0.0, "vy": 0.0, "wz": -1.0, "heading": 0.0,
    },
    #"forward_turn_left": {
    #    "vx": 1.0, "vy": 0.0, "wz": 0.5, "heading": 0.0,
    #},
    #"forward_turn_right": {
    #    "vx": 1.0, "vy": 0.0, "wz": -0.5, "heading": 0.0,
    #},
    #"diag_turn_left": {
    #    "vx": 0.5, "vy": 0.5, "wz": 0.5, "heading": 0,
    #},
    #"diag_turn_right": {
    #    "vx": 0.5, "vy": -0.5, "wz": -0.5, "heading": 0,
    #},
    "heading_90": {
        "vx": 0.0, "vy": 0.0, "wz": 0.0, "heading": np.pi / 2,
    },
    "heading_180": {
        "vx": 0.0, "vy": 0.0, "wz": 0.0, "heading": np.pi,
    },
    "heading_minus90": {
        "vx": 0.0, "vy": 0.0, "wz": 0.0, "heading": -np.pi / 2,
    },
    "stand": {
        "vx": 0.0, "vy": 0.0, "wz": 0.0, "heading": 0.0,
    },
}

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

    parser.add_argument('--save_depth_classifier_data', action='store_true', default=False, help="if with depth cam use to save depth image and rpy of all executed steps")
    parser.add_argument('--jit',            type=str, default='', help="path to a jit-scripted policy to load and swap with the trained policy (replaces JIT env var)")
    parser.add_argument('--curriculum', action='store_true', default=False, help="Load default curriculum")
    parser.add_argument('--test_terrain', type=str, default='random_uniform', help="current test_terrain")
    parser.add_argument('--no_depth_cam', action='store_true', default=False, help="disable test cam if available")
    parser.add_argument('--terrain_detector', type=str, default='', help="test a terrain detector")
    parser.add_argument('--command_test_suite', action='store_true', default=False, help="run a simple commnad test suite")
    
    return configure_runtime_device(parser.parse_args())

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
            "step_height": -0.2,
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
            "gap_size": 1.0,
            "platform_size": 3.0,
        },
        "pit": {
            "type": "terrain_utils.pit_terrain",
            "depth": 0.5,
            "platform_size": 3.0,
        },
        "multiple_high_platforms" : {
            "type": 'terrain_utils.multiple_high_platforms_terrain',
            "high_platform_height": 0.6,
            "high_platform_length": 1,
            "high_platform_width": 1.5,
            "high_platform_interval": 1.5,
        }
    }
    if args.num_envs:
        envs = args.num_envs
    else:
        envs =  min(env_cfg.env.num_envs, 100)
    task_name = args.task
    # override some parameters for testing
    # number of environments
    #env_cfg.init_state.pos = [0, 3, 2]
    env_cfg.env.num_envs = envs
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.init_state.yaw_random_scale = 0
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = env_cfg.env.num_envs
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(min(env_cfg.env.num_envs, envs)))
    # adjust parameters according to terrain type

    #env_cfg.terrain.vertical_scale = 0.1
    if args.curriculum:
        env_cfg.terrain.num_cols = sum(1 for x in env_cfg.terrain.terrain_proportions if x > 0)
        env_cfg.terrain.num_rows = 10 // env_cfg.terrain.num_cols
        env_cfg.terrain.border_size = 5.0
    elif env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.border_size = 5.0
        if args.test_terrain == "plane":
            if env_cfg.terrain.mesh_type == "heightfield":
                env_cfg.terrain.terrain_proportions = [0] * 10
            elif env_cfg.terrain.mesh_type == "trimesh":
                pass
                #TODO add patch to allow planes in issac
            env_cfg.terrain.terrain_kwargs = {}
        else:
            env_cfg.terrain.curriculum = False
            env_cfg.terrain.selected   = True
            if args.test_terrain == "baseline":
                env_cfg.terrain.terrain_kwargs = TERRAIN_CONFIGS["random_uniform"]
            else:
                env_cfg.terrain.terrain_kwargs = TERRAIN_CONFIGS[args.test_terrain]

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
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    
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

    # logger = ExpLogger(train_cfg.runner.exp_data_path)

    if args.terrain_detector:
        from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import DepthTerrainClassifier
        terrain_detector = DepthTerrainClassifier(pca_dim=4, num_prototypes=3)
        terrain_detector.load(args.terrain_detector)

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
        terrain_name_log = args.test_terrain
    
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
        joystick = Joystick(joystick_type=args.joystick_type)
    
    # env.commands[:, 0] = 0.5
    # env.commands[:, 1] = 0
    # env.commands[:, 2] = 0
    # env.commands[:, 3] = 0
    
    #if args.record_frames:
    #    env.simulator._floating_camera.start_recording()
    # interaction loop
    if args.command_test_suite:
        iter_count = len(list(command_tests.keys()))
        test_names = list(command_tests.keys())
    else:
        iter_count = 4.0

    for i in range(int(iter_count*env.max_episode_length)):
        if args.command_test_suite:
            current = i // env.max_episode_length
            if i % env.max_episode_length == 0:
                name = test_names[current]
                cmd = command_tests[name]
                print(f"Running test {name}, {cmd}")
            env.commands[:, 0] = cmd["vx"]
            env.commands[:, 1] = cmd["vy"]
            env.commands[:, 2] = cmd["wz"]
            env.commands[:, 3] = cmd["heading"]
        elif args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        elif i % env.max_episode_length == 0:
            env._resample_commands(torch.arange(env.num_envs))

        if args.save_depth_classifier_data:
            pass

        
        if args.jit and i == env.max_episode_length:
            print(f"swap to {-1}")
            policy.swap(-1)
        if args.jit and i == 2*env.max_episode_length:
            print(f"swap to {SWAP}")
            policy.swap(SWAP)

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
            actions = policy(obs_buf.detach(), obs_history.detach(), depth.detach())
            if policy1 is not None:
                print(f"Models diff: {torch.sum(torch.abs(actions - policy1(obs_buf, obs_history, depth)))}")
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos, depth = env.step(actions.detach())
            if args.save_depth_classifier_data:
                depth_images_log.append(env.depth_sensor_output.detach().cpu().clone())
                base_rpy_log.append(env.simulator._base_euler.detach().cpu().clone())
                base_ang_vel_log.append(env.simulator.base_ang_vel.detach().cpu().clone())
            if args.terrain_detector:
                _depth = env.depth_sensor_output[0].squeeze().detach().cpu()
                _euler = env.simulator._base_euler[0].detach().cpu().clone()
                _angve = env.simulator.base_ang_vel[0].detach().cpu().clone()
                predicted = terrain_detector.predict_depth(_depth, _euler, _angve)
                print(f"Label predicted for frame {predicted.label}")
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

        if env.cfg.terrain.terrain_proportions[TERRAIN_INDEX["gap"]]:
            difficulty = env.simulator.terrain_levels / env.cfg.terrain.num_rows
            dist = torch.norm(env.simulator.base_pos - env.simulator.env_origins, dim=-1)
            bound = env.cfg.terrain.platform_size/2 + eval(env.cfg.terrain.terrain_curriculum_difficulty["gap_size"]) + 1
            out = dist >= bound
            env_ids = out.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                env.simulator._terrain_levels[env_ids] = env.simulator._max_terrain_level + 1
                env.reset_idx(env_ids)
        
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

        logger.log_states(
            {
                'base_cmd':env.commands.detach().cpu().numpy().tolist(),
                'base_pose':env.simulator.base_pos.detach().cpu().numpy().tolist(),
                'base_rpy':env.simulator.base_euler.detach().cpu().numpy().tolist(),
                'q_actual':env.simulator.dof_pos.detach().cpu().numpy().tolist(),
                'base_lin_vel':env.simulator.base_lin_vel.detach().cpu().numpy().tolist(),
                'base_ang_vel':env.simulator.base_ang_vel.detach().cpu().numpy().tolist(),
                'dof_vel':env.simulator.dof_vel.detach().cpu().numpy().tolist(),
                'proj_grav':env.simulator.projected_gravity.detach().cpu().numpy().tolist(),
                'feet_pos':env.simulator.feet_pos.detach().cpu().numpy().tolist(),
                'failure':list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist())),
            })
    if args.save_depth_classifier_data:
        depth_images_tensor = torch.stack(depth_images_log, dim=0).squeeze(2)   # [T, num_envs, 1, H, W] (or however depth is shaped)
        base_rpy_tensor = torch.stack(base_rpy_log, dim=0)       # [T, num_envs, 4]
        base_ang_vel_tensor = torch.stack(base_ang_vel_log, dim=0)
        save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs_depth_data', train_cfg.runner.experiment_name)
        path = get_load_path(save_dir, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        os.makedirs(save_dir, exist_ok=True)

        save_name = (f"{train_cfg.runner.experiment_name}_{new}_capture.pt"
                    if new else f"{train_cfg.runner.experiment_name}_capture.pt")
        save_path = os.path.join(os.path.dirname(path), save_name)

        torch.save({
            "depth_images": depth_images_tensor,
            "depth_images_shape": tuple(depth_images_tensor.shape),
            "base_rpy": base_rpy_tensor,
            "base_rpy_shape": tuple(base_rpy_tensor.shape),
            "base_ang_vel": base_ang_vel_tensor,
            "base_ang_vel_shape": tuple(base_ang_vel_tensor.shape),
            "terrain_name": terrain_name_log,
        }, save_path)

        print(f"Saved depth images {tuple(depth_images_tensor.shape)}, "
          f"base rpy {tuple(base_rpy_tensor.shape)}, "
          f"base ang vel {tuple(base_ang_vel_tensor.shape)}, "
          f"and terrain name '{terrain_name_log}' to: {save_path}")

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
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
    elif "pact" in task_name:
        exporter = PolicyExporterPACT(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

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
    env.simulator._terrain_levels[:] = env.simulator._max_terrain_level + 1
    env.reset_idx(torch.arange(env.num_envs))
    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)
    policy1 = None

    if args.jit:
        policy1 = policy
        policy = torch.jit.load(args.jit,  map_location=args.gpu if not args.cpu else 'cpu')
        policy.swap(SWAP)
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
