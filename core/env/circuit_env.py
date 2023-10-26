from typing import Optional

import numpy as np
import tianshou
import  gymnasium as gym
from gymnasium.spaces import MultiBinary
from loguru import logger

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

simulator = AerSimulator()

# get gates from compiled circuit
def get_compiled_gate(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:

    try:
        compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout, backend=simulator)
    except:
        return -1
    return compiled_circuit.num_nonlocal_gates()

def make_obs_space():

    # low = np.array([0 for o in range(5)])
    # high = np.array([1 for o in range(5)])
    # shape = (5,)
    # logger.info('make obs sapce')

    space = MultiBinary([5,5])
    return space


#
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
#
class CircuitEnv(gym.Env):
    metadata = {"render_modes": ["human"],"render_fps": 1}
    @logger.catch()
    def __init__(self, render_mode: Optional[str] = None):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment

        # self.action_space  = spaces.Box(
        #         #set bounds of knobs
        #         low =  np.array([0, 0, 1]).astype(np.float32),
        #         high = np.array([+100, +100, +100]).astype(np.float32),
        #         dtype=np.float32
        #     )  # knob1,knob2,knob3...
        self.points = [(1,0),(1,1),(1,2),(1,3),(1,4)]
        self.adj = coordinate2adjacent(self.points)

        self.step_cnt = 1

       # self.spec.reward_threshold = 20000


        self.observation_space = make_obs_space()
        self.action_space =  self.make_action_space()

    def render(self, mode="human"):
        pass

    def reset(self, seed=0, return_info=False, options=None):
        # We need the following line to seed self.np_random
        #super().reset(seed=seed)

        # clean the render collection and add the initial frame
        #self.renderer.reset()
        #self.renderer.render_step()

        # reset db config
        observation = self._get_obs()
        return  observation

    def step(self, action):
        self.step_cnt+=1
        # if self.step_cnt <10 :
        #     logger.info('actions:'+str(action[:7]))
        # r = 0
        # for v in action:
        #     if v ==20.0 or v ==1.0:
        #         r -= 100
        #     else :
        #         r += v
        #
        # observation =  [random.randint(1,10) for i in range(33)] #self._get_obs()
        # reward = self._get_rewards()
        # info = self._get_info()
        # return observation, reward, False, info

        reward,observation = self._get_rewards(action)

        info = self._get_info()
        # add a frame to the render collection
        #self.renderer.render_step()

        done = False
        if reward == -1:
            done = True
        return observation, reward, done, info


    def render(self):
        print('render')
    def close(self):
        self._close_env()

    def _get_obs(self):
        return adjacency2matrix(coordinate2adjacent(self.points))

    def _get_info(self):
        return {"info":"this is info"}

    def _get_rewards(self,action):
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
            if action[i]:
                a.append(i)
                count += 1
        if count !=3:
            return -1
        res = get_compiled_gate(circuit,self.adj, a)
        if res ==-1:
            return res
        else:
            return 10-res

    '''
            There are two common use cases:

        * Identical bound for each dimension::
            >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
            Box(3, 4)

        * Independent bound for each dimension::
            >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
            Box(2,)
        '''

    def make_action_space(self):
        space = MultiBinary(5)
        return space
        # return gym.spaces.Box(low=0, high=1,shape=(1,4), dtype=np.int)

    def _close_env(self):
        logger.info('_close_env')