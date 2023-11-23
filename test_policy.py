import argparse
import os
import pprint
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, DDPGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from gymnasium import register
from tianshou.env import (
    ContinuousToDiscrete,
    DummyVectorEnv,
    MultiDiscreteToDiscrete,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
    VectorEnvNormObs,
)

from config import get_args

def test_policy():
    args = get_args()
    env = MultiDiscreteToDiscrete(gym.make(args.task))
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print(args.reward_threshold)

    # train_envs = gym.make(args.task)

    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: MultiDiscreteToDiscrete(gym.make(args.task)) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: MultiDiscreteToDiscrete(gym.make(args.task)) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(Actor(net, args.action_shape, device=None).to(args.device))
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.action_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_scaling=isinstance(env.action_space, Box),
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,

    )
    # log
    #log_path = os.path.join(args.logdir, args.task, "ppo")
    log_path = 'D:\workspace\data\early_stop\CircuitEnvTest-v2\ppo'
    policy.load_state_dict(torch.load(log_path+"\\policy.pth",map_location=torch.device('cpu')))
    env = MultiDiscreteToDiscrete(gym.make(args.task))
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=10, render=args.render)
    print(result)

if __name__ == '__main__':
    # 代码精简，action space 和 obs space 重构
    register(
        id='CircuitEnvTest-v2',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v2:CircuitEnvTest_v2',
        max_episode_steps=4000000,
    )
    args = get_args()
    kwargs = {
        'debug':True
    }

    env = MultiDiscreteToDiscrete(gym.make(args.task,**kwargs))
    #test_policy()