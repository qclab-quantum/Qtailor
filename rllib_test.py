import datetime
import random
import tempfile
import time
import math
import gymnasium as gym
from gymnasium import register
import numpy as np
import os

import ray
from ray import air, tune
from ray.air import CheckpointConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.registry import get_trainable_cls
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable
from ray import cloudpickle
from shared_memory_dict import SharedMemoryDict

from config import  ConfigSingleton
from temp.env.env_test_v3 import CircuitEnvTest_v3
from utils.benchmark import Benchmark
from utils.file_util import FileUtil

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
from config import args

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
def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

def load_checkpoint_from_path(checkpoint_to_load: Union[str, Path]) -> Dict:
    """Utility function to load a checkpoint Dict from a path."""
    checkpoint_path = Path(checkpoint_to_load).expanduser()
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    with checkpoint_path.open("rb") as f:
        return cloudpickle.load(f)

def train_policy():
    ray.init(num_gpus = 1)
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env = CircuitEnvTest_v3,env_config={"debug": False})
        .framework(args.framework)
        .rollouts(num_rollout_workers=args.num_rollout_workers)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #.resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=int(1))
    )
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    #每个 1 iter 都保存一次 checkpoint
    Checkpoint=  CheckpointConfig(checkpoint_frequency = args.checkpoint_frequency
                                  ,checkpoint_at_end=args.checkpoint_at_end)

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = args.rllib_lr
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
            #当reward 有提示，保存 checkpoint
            if result["episode_reward_mean"] > best_reward:
                best_reward = result["episode_reward_mean"]
                checkpoint = algo.save()
                print(f"New best reward: {best_reward}. Checkpoint saved to: {checkpoint}")
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop,checkpoint_config=Checkpoint,log_to_file=True),
        )
        results = tuner.fit()

        # ###################### evaluate start ######################
        print("Training completed. Restoring new Algorithm for action inference.")
        # Get the last checkpoint from the above training run.
        checkpoint = results.get_best_result().checkpoint
        test_result(checkpoint)
        #######################  evaluate end ######################

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

'''
checkpoint can be string or Checkpoint Object
'''
def test_result(checkpoint):
    algo = Algorithm.from_checkpoint(checkpoint)

    register(
        id='CircuitEnvTest-v3',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v3:CircuitEnvTest_v3',
        max_episode_steps=4000000,
    )

    # Create the env to do inference in.
    env = gym.make("CircuitEnvTest-v3")
    obs, info = env.reset()
    num_episodes = 0
    episode_reward = 0.0
#    while num_episodes < args.num_episodes_during_inference:

    content = ''
    while num_episodes < 1:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=None,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(a)
        info  = 'done = %r, reward = %r \n' % (done, reward)
        print(info)
        content += info
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            shape = int(math.sqrt(len(obs)))
            reshape_obs = np.array(obs).reshape(shape, shape)
            print('done = %r, reward = %r \n obs = \n {%r} ' % (done, reward,reshape_obs ))
            content += ('done = %r, reward = %r obs = \n %r ' % (done, reward, reshape_obs))
            print(f"Episode done: Total reward = {episode_reward}")
            #log to file
            #content += (f"Episode done: Total reward = {episode_reward}")
            rl,rl_qiskit,qiskit = Benchmark.depth_benchmark( reshape_obs, smd['qasm'], False)
            content += ('\n @rl = %r, rl_qiskit = %r, qiskit = %r@ '%(rl,rl_qiskit,qiskit))
            log2file(content)

            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0

    algo.stop()

def log2file(content):
    rootdir = FileUtil.get_root_dir()
    sep =os.path.sep
    path = rootdir+sep+'benchmark'+sep+'a-result'+sep+str(smd['qasm'])+'_'+str(args.log_file_id)+'.txt'
    FileUtil.write(path, content)

def get_qasm():
    qasm = [

        'ghz/ghz_indep_qiskit_10.qasm',
          'ghz/ghz_indep_qiskit_15.qasm',
        #'ghz\\ghz_indep_qiskit_25.qasm',
        #  'ghz/ghz_indep_qiskit_20.qasm',

    #     'ghz/ghz_indep_qiskit_30.qasm',
    #     'ghz/ghz_indep_qiskit_35.qasm',
    ]
    return qasm
if __name__ == "__main__":
    set_logger()
    import os
    os.environ["SHARED_MEMORY_USE_LOCK"] = '1'

    #print(f"Running with following CLI options: {args}")

    qasms = get_qasm()

    args = ConfigSingleton().get_config()
    for q in qasms:
        args.log_file_id = random.randint(0, 10000)

        smd = SharedMemoryDict(name='tokens',size=1024)
        smd['qasm'] = q

        print('run %r with id %r'%(smd['qasm'],args.log_file_id))

        train_policy()
        time.sleep(5)
    #test_result(checkpoint_path='C:/Users/Administrator/ray_results/PPO_2023-12-13_10-31-08/PPO_CircuitEnvTest_v3_ac724_00000_0_2023-12-13_10-31-08/checkpoint_000000')
