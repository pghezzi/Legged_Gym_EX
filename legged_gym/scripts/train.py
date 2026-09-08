import os


from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import shutil
import time
import torch

from legged_gym.scripts.expand_config import reconstruct_config_file

def train(args):
    post_specialist_started = (
        time.perf_counter() if "distill" in args.task.lower() else None
    )
    if post_specialist_started is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning')
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if (post_specialist_started is not None
            and hasattr(ppo_runner, "_post_specialist_started")):
        # Include environment construction in the distillation-only accounting
        # boundary without affecting any runner's training behavior.
        ppo_runner._post_specialist_started = post_specialist_started
    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if env_cfg.asset.name == args.task:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+"_config.py")
    else:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+"_config.py")
    shutil.copy(robot_file_path, log_dir)
    shutil.copy(robot_config_path, log_dir)
    #pablo paranoia
    simulator_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "simulator", f"{SIMULATOR}_simulator.py")
    shutil.copy(simulator_path, log_dir)
    if "lora" in log_dir:
        shutil.copy(robot_file_path.replace("_lora", ""), log_dir)
        lora_handler_path = os.path.join(LEGGED_GYM_ROOT_DIR, "rsl_rl", "utils", "LoRA.py")
        shutil.copy(lora_handler_path, log_dir)
    output_path = os.path.join(log_dir, f"{args.task}_expanded_config.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(reconstruct_config_file(env_cfg.__class__, train_cfg.__class__))
    
    # Start training session
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
