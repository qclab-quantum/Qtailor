import datetime
import random
import time
import gymnasium as gym
from gymnasium import register
import os
import ray
from ray import air, tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from shared_memory_dict import SharedMemoryDict

from config import  ConfigSingleton
from temp.env.env_test_v5 import CircuitEnvTest_v5
from temp.env.env_test_v7 import CircuitEnvTest_v7
from utils.benchmark import Benchmark
from utils.csv_util import CSVUtil
from utils.file_util import FileUtil
from utils.graph_util import GraphUtil
from io import StringIO
import contextlib
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
from rllib_helper import set_logger, new_csv, get_qasm, parse_tensorboard

csv_path = ''
text_path=''
datetime_str =''
tensorboard=''
args = None
def train_policy():
    #os.environ.get("RLLIB_NUM_GPUS", "1")
    ray.shutdown()
    ray.init(num_gpus = 1,local_mode=args.local_mode)
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env = CircuitEnvTest_v5)
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
    # Checkpoint_config=  CheckpointConfig(checkpoint_frequency = args.checkpoint_frequency
    #                               ,checkpoint_at_end=args.checkpoint_at_end)

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

        #evaluate
        print("Training completed")
        return results


def test_result(checkpoint):

    algo = Algorithm.from_checkpoint(checkpoint)
    env_id = "CircuitEnvTest-v"+str(args.env_version)
    smd = SharedMemoryDict(name='tokens', size=1024)
    smd['evaluate'] = True
    smd['debug'] = True
    register(
        id=env_id,
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v'+str(args.env_version)+':'+'CircuitEnvTest_v'+str(args.env_version),
        max_episode_steps=4000000,
    )

    # Create the env to do inference in.
    env = gym.make(env_id)
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
        obs, reward, done, truncated, info = env.step(a)
        print('done = %r, reward = %r \n' % (done, reward))
        episode_reward += reward

        # Is the episode `done`? -> Reset.
        if done:
            # shape = int(math.sqrt(len(obs)))
           #reshape_obs = np.array(obs).reshape(shape, shape)
            reshape_obs = GraphUtil.restore_from_1d_array(obs)
           #  print('info = ', info)
           #  obs = info['matrix']
           #  reshape_obs = info['matrix']
            print('done = %r, reward = %r \n obs = \n {%r} ' % (done, reward,reshape_obs ))

            print(f"Episode done: Total reward = {episode_reward}")
            #log to file
            rl,qiskit,mix = Benchmark.depth_benchmark( csv_path,reshape_obs, smd['qasm'], False)

            if not isinstance(checkpoint,str):
                checkpoint = checkpoint.path
            log2file(rl, qiskit, mix,  obs,args.stop_iters, checkpoint,)

            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0

    smd.shm.close()
    smd.shm.unlink()
    algo.stop()

def log2file(rl, qiskit, mix,  result,iter_cnt, checkpoint):
    # rootdir = FileUtil.get_root_dir()
    # sep =os.path.sep
    # path = rootdir+sep+'benchmark'+sep+'a-result'+sep+str(smd['qasm'])+'_'+str(args.log_file_id)+'.txt'
    # FileUtil.write(path, content)
    smd = SharedMemoryDict(name='tokens', size=1024)
    data = [datetime_str,smd['qasm'],rl, qiskit, mix,  result,iter_cnt, checkpoint,tensorboard]
    CSVUtil.append_data(csv_path,[data])


def train():
    global csv_path
    global datetime_str
    global text_path
    global tensorboard
    csv_path = new_csv(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    qasms = get_qasm()

    sep = '/'

    for q in qasms:

        args.log_file_id = random.randint(1000, 9999)
        smd = SharedMemoryDict(name='tokens', size=1024)
        smd['qasm'] = q

        datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        text_path = FileUtil.get_root_dir() + sep + 'benchmark' + sep + 'a-result' + sep + q + '_' + datetime_str + '.txt'

        # Create a StringIO object to redirect the console output
        output = StringIO()
        with contextlib.redirect_stdout(output):
            results = train_policy()

        strings = output.getvalue()

        FileUtil.write(text_path, strings)
        tensorboard = parse_tensorboard(strings)

        checkpoint = results.get_best_result().checkpoint
        test_result(checkpoint)

        output.truncate(0)
        output.close()

        time.sleep(5)
def test():
    checkpoint = r'D:\workspace\data\AblationStudy\PPO_2024-01-02_20-25-47\PPO_CircuitEnvTest_v5_05cbb_00000_0_2024-01-02_20-25-47\checkpoint_000000'
    new_csv(datetime_str)

    smd = SharedMemoryDict(name='tokens', size=1024)
    smd['qasm'] = 'qnn/qnn_indep_qiskit_8.qasm'
    try:
        test_result(checkpoint)
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
    # 设置环境变量
    #os.environ['TUNE_RESULT_DIR'] = 'd:/tensorboard'
    #给 SharedMemoryDict 加锁
    os.environ["SHARED_MEMORY_USE_LOCK"] = '1'
    #test()
    #test_checkpoint()
    train()





