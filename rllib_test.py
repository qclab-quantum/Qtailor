"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from temp.env.env_test_v3 import CircuitEnvTest_v3
from train_policy import register_env

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=1.5, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=config.worker_index * config.num_workers)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        return [self.cur_pos], {}

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = truncated = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return (
            [self.cur_pos],
            random.random() * 2 if done else -0.1,
            done,
            truncated,
            {},
        )

def set_logger():
    import warnings
    warnings.simplefilter('ignore')
    import logging

    # First, get the handle for the logger you want to modify
    ray_data_logger = logging.getLogger("ray.data")
    ray_tune_logger = logging.getLogger("ray.tune")
    ray_rllib_logger = logging.getLogger("ray.rllib")
    ray_train_logger = logging.getLogger("ray.train")
    ray_serve_logger = logging.getLogger("ray.serve")

    # Modify the ray.data logging level
    ray_data_logger.setLevel(logging.ERROR)
    ray_tune_logger.setLevel(logging.ERROR)
    ray_rllib_logger.setLevel(logging.ERROR)
    ray_train_logger.setLevel(logging.ERROR)
    ray_serve_logger.setLevel(logging.ERROR)
if __name__ == "__main__":

    set_logger()
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env = CircuitEnvTest_v3,env_config={"debug": False})
        .framework(args.framework)
        .rollouts(num_rollout_workers=1)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()
        ###evaluate start
        print("Training completed. Restoring new Algorithm for action inference.")
        # Get the last checkpoint from the above training run.
        checkpoint = results.get_best_result().checkpoint
        # Create new Algorithm and restore its state from the last checkpoint.
        algo = Algorithm.from_checkpoint(checkpoint)

        register_env()
        # Create the env to do inference in.
        env = gym.make("CircuitEnvTest-v3")
        obs, info = env.reset()

        num_episodes = 0
        episode_reward = 0.0

        while num_episodes < args.num_episodes_during_inference:
            # Compute an action (`a`).
            a = algo.compute_single_action(
                observation=obs,
                explore=args.explore_during_inference,
                policy_id="default_policy",  # <- default value
            )
            # Send the computed action `a` to the env.
            obs, reward, done, truncated, _ = env.step(a)
            episode_reward += reward
            # Is the episode `done`? -> Reset.
            if done:
                print(f"Episode done: Total reward = {episode_reward}")
                obs, info = env.reset()
                num_episodes += 1
                episode_reward = 0.0

        algo.stop()
        ###evaluate end
        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()