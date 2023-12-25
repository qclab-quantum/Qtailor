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
from ray.tune import ResultGrid
from ray.tune.trainable import trainable
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.registry import get_trainable_cls
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable
from ray import cloudpickle
from shared_memory_dict import SharedMemoryDict

from config import  ConfigSingleton
from temp.env.env_test_v3 import CircuitEnvTest_v3
from temp.env.env_test_v4 import CircuitEnvTest_v4
from utils.benchmark import Benchmark
from utils.csv_util import CSVUtil
from utils.file_util import FileUtil
from utils.graph_util import GraphUtil

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

csv_path = ''
datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
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
args = None
def train_policy():
    #os.environ.get("RLLIB_NUM_GPUS", "1")
    ray.init(num_gpus = 1,local_mode=args.local_mode)
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env = CircuitEnvTest_v4,env_config={"debug": False})
        .framework(args.framework)
        .rollouts(num_rollout_workers=args.num_rollout_workers
                  #,num_envs_per_worker=5
                  #,remote_worker_envs=True
                  )
        .resources(num_gpus=1)
    )
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    '''
    Checkpoints are py-version specific, but can be converted to be version independent
    https://docs.ray.io/en/latest/rllib/rllib-saving-and-loading-algos-and-policies.html
    '''
    Checkpoint_config=  CheckpointConfig(checkpoint_frequency = args.checkpoint_frequency
                                  ,checkpoint_at_end=args.checkpoint_at_end)

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = [(0, 0.001), (1e6, 0.0001), (2e6, 0.00005)]

        algo = None
        #resuse from check point
        if args.resume:
            algo = Algorithm.from_checkpoint(args.checkpoint)
        else:
        # new algo
            algo = config.build()
        # run manual training loop and print results after each iteration
        TrainingResult = None
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

            if result["episode_reward_mean"] > -100:
                best_reward = result["episode_reward_mean"]
                TrainingResult = algo.save()
                print(f"New best reward: {best_reward}. Checkpoint saved to: {TrainingResult}")

        test_result(TrainingResult.checkpoint.path)
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop
                                     # ,checkpoint_config=Checkpoint_config
                                     # ,log_to_file=True
                                     ),

        )
        results = tuner.fit()
        #analyze_result(results)

        #evaluate
        print("Training completed")
        checkpoint = results.get_best_result().checkpoint
        test_result(checkpoint)

    ray.shutdown()

def analyze_result(results:ResultGrid):
    ## result analysis
    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.path  # Get best trial's result directory
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    # Get a dataframe with the last results for each trial
    df_results = results.get_dataframe()

    # Get a dataframe of results for a specific score or mode
    df = results.get_dataframe(filter_metric="score", filter_mode="max")
    ##
    # print(best_config)
    # print('====================')
    # print(best_metrics)

def test_result(checkpoint):

    algo = Algorithm.from_checkpoint(checkpoint)

    register(
        id='CircuitEnvTest-v4',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v4:CircuitEnvTest_v4',
        max_episode_steps=4000000,
    )

    # Create the env to do inference in.
    env = gym.make("CircuitEnvTest-v4")
    obs, info = env.reset()
    num_episodes = 0
    episode_reward = 0.0
#    while num_episodes < args.num_episodes_during_inference:

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
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            # shape = int(math.sqrt(len(obs)))
            # reshape_obs = np.array(obs).reshape(shape, shape)
            reshape_obs = GraphUtil.restore_from_1d_array(obs)
            print('done = %r, reward = %r \n obs = \n {%r} ' % (done, reward,reshape_obs ))

            print(f"Episode done: Total reward = {episode_reward}")
            #log to file
            smd = SharedMemoryDict(name='tokens', size=1024)
            rl,qiskit,rl_qiskit = Benchmark.depth_benchmark( csv_path,reshape_obs, smd['qasm'], False)

            if not isinstance(checkpoint,str):
                checkpoint = checkpoint.path
            log2file(rl, qiskit, rl_qiskit,  obs,args.stop_iters, checkpoint)

            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0

    algo.stop()

def log2file(rl, qiskit, rl_qiskit,  result,iter_cnt, checkpoint):
    # rootdir = FileUtil.get_root_dir()
    # sep =os.path.sep
    # path = rootdir+sep+'benchmark'+sep+'a-result'+sep+str(smd['qasm'])+'_'+str(args.log_file_id)+'.txt'
    # FileUtil.write(path, content)
    smd = SharedMemoryDict(name='tokens', size=1024)
    data = [datetime_str,smd['qasm'],rl, qiskit, rl_qiskit,  result,iter_cnt, checkpoint]
    CSVUtil.append_data(csv_path,[data])
def new_csv():
    rootdir = FileUtil.get_root_dir()
    sep = '/'
    global csv_path
    csv_path = rootdir + sep + 'benchmark' + sep + 'a-result' + sep + datetime_str + '.csv'
    print(csv_path)
    CSVUtil.write_data(csv_path,
                       [['datetime', 'qasm', 'rl', 'qiskit', 'rl_qiskit', 'result', 'iter', 'checkpoint','remark', ]])
def get_qasm():
    qasm = [
       'qnn/qnn_indep_qiskit_6.qasm',
       'qnn/qnn_indep_qiskit_7.qasm',
       'qnn/qnn_indep_qiskit_8.qasm',
       'qnn/qnn_indep_qiskit_9.qasm',
       'qnn/qnn_indep_qiskit_10.qasm',
    ]
    return qasm
def train():
    new_csv()
    qasms = get_qasm()

    smd = SharedMemoryDict(name='tokens', size=1024)
    for q in qasms:
        args.log_file_id = random.randint(1000, 9999)
        smd['qasm'] = q
        # print('run %r with id %r'%(smd['qasm'],args.log_file_id))
        train_policy()
        time.sleep(5)
    smd.shm.close()
    smd.shm.unlink()
def test():
    checkpoint = r'C:/Users/Administrator/ray_results/PPO_2023-12-25_12-58-19/PPO_CircuitEnvTest_v4_39103_00000_0_2023-12-25_12-58-19/checkpoint_000000'
    new_csv()

    smd = SharedMemoryDict(name='tokens', size=1024)
    smd['qasm'] = 'qnn/qnn_indep_qiskit_3.qasm'
    try:
        test_result(checkpoint)
        smd.shm.close()
        smd.shm.unlink()
    except Exception as e:
        print(e)
    finally:
        smd.shm.close()
        smd.shm.unlink()
def test_checkpoint():
    smd = SharedMemoryDict(name='tokens', size=1024)
    smd['qasm'] = 'qnn/qnn_indep_qiskit_3.qasm'
    try:
        checkpoint = r'C:\Users\Administrator\AppData\Local\Temp\tmppvaizpqo'
        algo = Algorithm.from_checkpoint(checkpoint)
        smd.shm.close()
        smd.shm.unlink()
    except Exception as e:
        print(e)
    finally:
        smd.shm.close()
        smd.shm.unlink()
if __name__ == "__main__":
    args = ConfigSingleton().get_config()
    set_logger()
    #给 SharedMemoryDict 加锁
    os.environ["SHARED_MEMORY_USE_LOCK"] = '1'
    test()
    train()



