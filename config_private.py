import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--task", type=str, default='circuit')
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256,256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)

    #collect
    parser.add_argument("--start-timesteps", type=int, default=25000)#collect 的 n_step 收集的step的数量
    #OffpolicyTrainer
    parser.add_argument("--max-epoch", type=int, default=20000) #最大
    parser.add_argument("--step-per-epoch", type=int, default=10240)
    parser.add_argument("--step-per-collect", type=int, default=10240)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=20000*10240)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=20)
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