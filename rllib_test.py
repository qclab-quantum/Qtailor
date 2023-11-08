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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

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
    "--stop-iters", type=int, default=20000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=8, help="Reward at which we stop training."
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




import gymnasium as gym
import numpy
import numpy as np

from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from loguru import logger
simulator = AerSimulator()

def get_compiled_gate(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:

    try:
        compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout, backend=simulator)
        return compiled_circuit.num_nonlocal_gates()
    except:
        return -1


# get adj  from coordinate
def coordinate2adjacent(points):
    import math
    point_dict = {i: points[i] for i in range(len(points))}
    adjacency_dict = {}

    for i, point in point_dict.items():
        adjacent_points = []
        for j, other_point in point_dict.items():
            if i != j:
                if math.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2) == 1:
                    adjacent_points.append(j)
        adjacency_dict[i] = adjacent_points

    #transform adjacency_dict to qiskit format
    res = []
    for k in adjacency_dict:
        v = adjacency_dict.get(k)
        for node in v:
            res.append([k,node])
    # return adjacency_dict
    return res

def adjacency2matrix(adj_list):
    max_index = max(max(pair) for pair in adj_list)
    matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

    for pair in adj_list:
        matrix[pair[0]][pair[1]] = 1
        matrix[pair[1]][pair[0]] = 1

    return matrix

class CircuitEnvTest(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # qpu Topology
        self.points = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),(1, 5),(1, 6),(1, 7),(1, 9),(1, 10)]
        self.adj = coordinate2adjacent(self.points)
        self.step_cnt = 1

        self.observation_space = self.make_obs_space()
        self.action_space = MultiDiscrete([2,2,2,2,2,2,2,2,2,2])
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """


    def make_obs_space(self):
       # space = MultiBinary([10, 10])
        space = MultiDiscrete(np.array([2] * 100))

        return space
    def _get_info(self):
        return {'info':'this is info'}



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    '''
    Returns:
                observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                    An example is a numpy array containing the positions and velocities of the pole in CartPole.
                reward (SupportsFloat): The reward as a result of taking the action.
                terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                    which can be positive or negative. An example is reaching the goal state or moving into the lava from
                    the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
                truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                    Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                    Can be used to end the episode prematurely before a terminal state is reached.
                    If true, the user needs to call :meth:`reset`.
                info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                    This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                    hidden from observations, or individual reward terms that are combined to produce the total reward.
                    In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                    however this is deprecated in favour of returning terminated and truncated variables.
                done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                    return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                    A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                    a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
    '''
    def step(self, action):
        self.step_cnt = self.step_cnt+1
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        reward,observation = self._get_rewards(action)
        info = self._get_info()

        # if self.step_cnt % 100 == 0:
        #     print('step = ',self.step_cnt)

        terminated = False
        truncated = False
        if reward <= 0:
            terminated = True
        if reward ==7:
            terminated = True
            #print('action=',action)
        #print(reward)
        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')
    # %%
    # Close
    # ~~~~~
    #
    # The ``close`` method should close any open resources that were used by
    # the environment. In many cases, you don’t actually have to bother to
    # implement this method. However, in our example ``render_mode`` may be
    # ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        self._close_env()

    def _get_obs(self):
        obs = np.array(adjacency2matrix(coordinate2adjacent(self.points)))
        return obs.flatten("C")

    def _get_info(self):
        return {"info":"this is info"}

    def _get_rewards(self,action):

        obs = self._get_obs()
        reward = -1
        circuit = QuantumCircuit(3, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.measure([0, 1], [0, 1])

        a = []
        count = 0
        for i in range(len(action)):
            if action[i]==1:
                a.append(i)
                count += 1
        if count !=3:
            reward =  -abs(abs(count)-3)
        else:
          res = get_compiled_gate(circuit,self.adj, a)
          if res != -1:
              reward = 10 - res
        return reward,obs

    def _close_env(self):
        logger.info('_close_env')


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    # config = (get_trainable_cls(args.run)
    #     .get_default_config()
    #     # or "corridor" if registered above
    #     .environment(CircuitEnvTest, env_config={})
    #     .framework(args.framework)
    #     .rollouts(num_rollout_workers=16,num_envs_per_worker=20,rollout_fragment_length = 'auto')
    #
    #     .training(model={"fcnet_hiddens": [32,64, 128,64,32]},gamma =0.9 )
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     #.resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    # )

    config = (
        PPOConfig()
        .rollouts(num_rollout_workers=5,
                  #num_envs_per_worker=20,
                  rollout_fragment_length = 'auto'
                  )
        .resources(num_gpus=0)
        .framework(args.framework)
        .environment(CircuitEnvTest, env_config={})
        .training(model={"fcnet_hiddens": [32, 64, 128, 64, 32]}, gamma=0.9)

    )
    # use fixed learning rate instead of grid search (needs tune)
    config.lr = 1e-4

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # manual training with train loop using PPO and fixed learning rate
    if args.run != "PPO":
        raise ValueError("Only support --run PPO with --no-tune.")
    print("Running manual train loop without Ray Tune.")

    algo = config.build()
    # run manual training loop and print results after each iteration
    for _ in range(args.stop_iters):
        result = algo.train()
        print('===============iter %r start ====================='% _)
        print(pretty_print(result))
        print('===============iter %r end ====================='% _)
        #print(result['episode_reward_mean'],'  ',result['timers'])
        # stop training of the target train steps or reward are reached
        if _ % 5==0:
            checkpoint_dir = algo.save().checkpoint.path
        if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
        ):
            break

    algo.stop()
    ray.shutdown()