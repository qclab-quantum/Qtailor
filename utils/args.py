import torch
import argparse

class args:
    def __init__(self)-> None:
        super().__init__()

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, default="CartPole-v0")
        parser.add_argument("--seed", type=int, default=1626)
        parser.add_argument("--buffer-size", type=int, default=50000)
        parser.add_argument("--actor-lr", type=float, default=3e-4)
        parser.add_argument("--critic-lr", type=float, default=3e-4)
        parser.add_argument("--alpha-lr", type=float, default=3e-4)
        parser.add_argument("--noise_std", type=float, default=1.2)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--auto_alpha", type=int, default=1)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--epoch", type=int, default=20)
        parser.add_argument("--step-per-epoch", type=int, default=12000)
        parser.add_argument("--step-per-collect", type=int, default=5)
        parser.add_argument("--update-per-step", type=float, default=0.2)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
        parser.add_argument("--training-num", type=int, default=30)
        parser.add_argument("--test-num", type=int, default=10)
        parser.add_argument("--logdir", type=str, default="log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        return parser.parse_args()

