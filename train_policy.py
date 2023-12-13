import os
import pprint
import time
from tianshou.utils import WandbLogger
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
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

kwargs = {
    'debug': False
}


def train_ppo(args=get_args()):
    env = MultiDiscreteToDiscrete(gym.make(args.task,**kwargs))
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # train_envs = gym.make(args.task)

    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([lambda: MultiDiscreteToDiscrete(gym.make(args.task,**kwargs) )for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: MultiDiscreteToDiscrete(gym.make(args.task,**kwargs)) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        print('cuda is available \n')
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

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, "ppo")
    # writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer)

    #wandb logger
    logger = WandbLogger(project = 'CircuitEnvTest_v3',name  = '2023.12.2204_ppo_1', run_id = 'ppo1')
    logger.load(SummaryWriter(log_path))

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold


    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    print('train result = \n',result)
   #assert stop_fn(result["rews"].mean())

    if __name__ == "__main__":
        #pprint.pprint(result)
        # Let's watch its performance!
        env = MultiDiscreteToDiscrete(gym.make(args.task,**kwargs))
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=5, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        pprint.pprint(result)

def register_env():
    # 最简单的环境
    # register(
    #     id='CircuitEnvTest-v0',
    #     # entry_point='core.envs.circuit_env:CircuitEnv',
    #     entry_point='temp.env.env_test_v0:CircuitEnvTest_v0',
    #     max_episode_steps=2000000,
    # )

    #田字格 5 比特
    # register(
    #     id='CircuitEnvTest-v1',
    #     # entry_point='core.envs.circuit_env:CircuitEnv',
    #     entry_point='temp.env.env_test_v1:CircuitEnvTest_v1',
    #     max_episode_steps=2000000,
    # )

    #代码精简，action space 和 obs space 重构
    # register(
    #     id='CircuitEnvTest-v2',
    #     # entry_point='core.envs.circuit_env:CircuitEnv',
    #     entry_point='temp.env.env_test_v2:CircuitEnvTest_v2',
    #     max_episode_steps=4000000,
    # )
    register(
        id='CircuitEnvTest-v3',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v3:CircuitEnvTest_v3',
        max_episode_steps=4000000,
    )
def train():
    # Start the timer
    start_time = time.time()

    train_ppo()

    #获取执行时间
    end_time = time.time()
    runtime = round(end_time - start_time, 3)
    print("Function runtime:", runtime, "seconds")
if __name__ == "__main__":
    register_env()
    train()
