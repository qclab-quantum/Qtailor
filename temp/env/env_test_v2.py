import gymnasium as gym
import numpy
import numpy as np

from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
simulator = AerSimulator()

from utils.circuit_util import CircutUtil as cu
warnings.filterwarnings("ignore")
class CircuitEnvTest_v2(gym.Env):
    def __init__(self, render_mode=None, size=5):
        self.points = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),(1, 5),(1, 6),(1, 7),(1, 9),(1, 10)]
        self.adj = cu.coordinate2adjacent(self.points)
        self.step_cnt = 1

        self.observation_space = self.make_obs_space()
        self.action_space = MultiDiscrete([9,9])
        self.obs = [0,1,2,3,4,5,5,5,5]
        # obs[i] == qubit_nums 说明该位置为空，
        self.qubit_nums = 5
        self.flag = self.qubit_nums
        #上个动作获取到的score
        self.last_score = None

    def make_obs_space(self):
        space = MultiDiscrete(np.array([[6] * 9] ))

        return space
    def _get_info(self):
        return {'info':'this is info'}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.obs = [0,1,2,3,4,5,5,5,5]
        info = self._get_info()
        return self.obs, info

    def step(self, action):
        self.step_cnt = self.step_cnt+1
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward,observation = self._get_rewards(action)
        info = self._get_info()

        terminated = False
        truncated = False
        if reward <= 0:
            terminated = True
        if reward ==6:
            terminated = True
        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _get_obs(self):
        #obs = np.array(cu.adjacency2matrix(cu.coordinate2adjacent(self.points)))
        return self.obs

    def _get_info(self):
        return {"info":"this is info"}

    def can_switch(self,position_1,position_2):
        if position_1 == position_2:
            return False
        #源位置为空
        if self.obs[position_1] == self.flag:
            return False
        #目标位置已有
        elif self.obs[position_2] != self.flag:
            return False
        return True
    def _get_rewards(self,action):


        position_1 = action[0]
        position_2 = action[1]

        reward = -10
        #try switch the positon
        if self.can_switch(position_1,position_2):
           # 位置 2 的值设置为1
           self.obs[position_2]=self.obs[position_1]
           #源位置设置为空 (flag)
           self.obs[position_1]=self.flag

           circuit = QuantumCircuit(5)

           circuit.cx(0, 1)
           circuit.cx(0, 2)
           circuit.cx(0, 3)
           circuit.cx(0, 4)
           qr = circuit.qubits
           # 全连接
           adj = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1],
                  [2, 5], [3, 0], [3, 4], [3, 6], [4, 1], [4, 3], [4, 5],
                  [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 4],
                  [7, 6], [7, 8], [8, 5], [8, 7]]

           layout = [None] * len(self.obs)
           for i in range(len(self.obs)):
               v = self.obs[i]
               if  v != self.flag:
                   layout[i] = qr[v]

           # score 越低越好
           score = cu.get_circuit_score(circuit, adj, layout)
           if self.last_score is None:
               reward = 0
           elif score > 0:
               if score > self.last_score:
                   reward = 1
               elif score < self.last_score:
                   reward = -1.5
               else:
                   reward = -1.1
           else:
               reward = -10

           self.last_score = score

        return reward,self.obs

    def _close_env(self):
        logger.info('_close_env')


if __name__ == '__main__':
    pass