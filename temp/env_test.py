import argparse
import datetime
import os
import pprint
import gymnasium as gym
import numpy as np
import torch
from gymnasium import register
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
import loguru
import core
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='MysqlDB-v0')
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    #parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=True,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )


    return parser.parse_args()

def make_env(task,seed,training_num,test_num):
    env = gym.make(task)
    train_envs = DummyVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
    #env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    return env, train_envs, test_envs

if __name__ == "__main__":

    register(
        id='qcrlenv-v0',
        entry_point='core.env.circuit_env:CircuitEnv',
        max_episode_steps=30000,
    )
    env = gym.make('qcrlenv-v0')

    obs = env.reset()
    print(obs)
    print(env.action_space)
    print(env.observation_space)