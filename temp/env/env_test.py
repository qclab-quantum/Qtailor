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


# get adj table from coordinate
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
    matrix = [[False] * (max_index + 1) for _ in range(max_index + 1)]

    for pair in adj_list:
        matrix[pair[0]][pair[1]] = True
        matrix[pair[1]][pair[0]] = True

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
        space = MultiBinary([10, 10])

        return space
    def _get_info(self):
        return 'info'



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

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
            truncated = True
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
        return adjacency2matrix(coordinate2adjacent(self.points))

    def _get_info(self):
        return {"info":"this is info"}

    def _get_rewards(self,action):

        obs = self._get_obs()
        reward = -1

        circuit = QuantumCircuit(3, 2)
        # Add a H gate on qubit 0
        circuit.h(0)

        # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        # Map the quantum measurement to the classical bits
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

        # return gym.spaces.Box(low=0, high=1,shape=(1,4), dtype=np.int)

    def _close_env(self):
        logger.info('_close_env')


if __name__ == '__main__':
    print(MultiBinary([10, 10]).sample())