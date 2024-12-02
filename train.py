import numpy as np
import os
from datetime import datetime
from configs.go2_config import Go2RoughCfg, Go2RoughCfgPPO


import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry

def train(args):
    args.sim_device = "cuda:0"
    args.rl_device = args.sim_device
    args.headless = True
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("go2",LeggedRobot,Go2RoughCfg(),Go2RoughCfgPPO())
    args = get_args()
    train(args)
