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

class CircuitEnvTest_v0(gym.Env):
    def __init__(self, render_mode=None, size=5):
        self.points = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),(1, 5),(1, 6),(1, 7),(1, 9),(1, 10)]
        self.adj = coordinate2adjacent(self.points)
        self.step_cnt = 1

        self.observation_space = self.make_obs_space()
        self.action_space = MultiDiscrete([2,2,2,2,2,2,2,2,2,2])

    def make_obs_space(self):
       # space = MultiBinary([10, 10])
        space = MultiDiscrete(np.array([[2] * 10] * 10))

        return space
    def _get_info(self):
        return {'info':'this is info'}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_cnt = self.step_cnt+1
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward,observation = self._get_rewards(action)
        info = self._get_info()

        terminated = False
        truncated = False
        if reward <= 0:
            terminated = True
        if reward ==7:
            terminated = True
        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _get_obs(self):
        obs = np.array(adjacency2matrix(coordinate2adjacent(self.points)))
        return obs

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


if __name__ == '__main__':
    print(np.array([[2] * 10] * 10))