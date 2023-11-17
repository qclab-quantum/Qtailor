import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--task", type=str, default="CartPole-v0")
    #parser.add_argument("--task", type=str, default="MountainCar-v0")
    parser.add_argument("--task", type=str, default="CircuitEnvTest-v2")
    parser.add_argument("--reward-threshold", type=float, default=2)
    parser.add_argument("--seed", type=int, default=1996)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
    parser.add_argument("--step-per-collect", type=int, default=2000)
    parser.add_argument("--repeat-per-collect", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[32,64,128,128,64,32,16])
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rew-norm", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)

    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.4)
    parser.add_argument("--max-epoch", type=int, default=2000000)  # 最大
    parser.add_argument("--n-step", type=int, default=1024)
    return parser.parse_known_args()[0]