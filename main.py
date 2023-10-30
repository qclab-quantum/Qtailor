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


# !/usr/bin/envs python3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='MysqlDB-v0')
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256,256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=20000)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
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
    #envs.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    return env, train_envs, test_envs

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
    net_c = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
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
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            100,#episode_per_test
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()

    #test_envs.seed(args.seed)
    #test_collector.reset()
    #result = test_collector.collect(n_episode=100, render=args.render)
    result = test_collector.collect(n_step=args.start_timesteps, random=True, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":

    register(
        id='CircuitEnvTest',
       # entry_point='core.envs.circuit_env:CircuitEnv',
         entry_point='temp.env.env_test:CircuitEnvTest',
        max_episode_steps=30000,
    )
    # env = gym.make('CircuitEnvTest')
    # obs = env.reset()
    #wrapped_env = FlattenObservation(envs)
    #envs.reset(seed=0, return_info=False, options=None)

    test_ddpg()
