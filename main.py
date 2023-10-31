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
from config_private import get_args
import time
# !/usr/bin/envs python3


def make_env(task,seed,training_num,test_num):
    env = gym.make(task)
    train_envs = DummyVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
    #envs.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    return env, train_envs, test_envs
def on_train_epoch(num_epoch: int, step_idx: int):
    print('num_epoch:',num_epoch)

def test_ddpg(args=get_args()):
    print('training_num={},test_num={}'.format(args.training_num,args.test_num))
    env, train_envs, test_envs = make_env('CircuitEnvTest', 12345, args.training_num, args.test_num )

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(
        net_a, args.action_shape, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c = Net(args.state_shape,args.action_shape,hidden_sizes=args.hidden_sizes,concat=True,device=args.device,)
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        #buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ddpg"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.max_epoch,#每次训练的最大 epoch
            args.step_per_epoch,#每个epoch中收集的总的 transitions
            args.step_per_collect,#每次网络更新要收集的transitions
            10000,#episode_per_test #在evaluation时，执行的episode的数量
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
            train_fn = on_train_epoch,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()

    test_envs.seed(args.seed)
    test_collector.reset()
    #result = test_collector.collect(n_episode=100, render=args.render)
    result = test_collector.collect(n_step=args.start_timesteps, random=True, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    start_time = time.time()
    register(
        id='CircuitEnvTest',
       # entry_point='core.envs.circuit_env:CircuitEnv',
         entry_point='temp.env.env_test:CircuitEnvTest',
        max_episode_steps=200000000,
    )
    # env = gym.make('CircuitEnvTest')
    # obs = env.reset()
    #wrapped_env = FlattenObservation(envs)
    #envs.reset(seed=0, return_info=False, options=None)

    test_ddpg()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
