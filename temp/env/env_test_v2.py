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

        self.observation_space = self.make_obs_space()
        self.action_space = MultiDiscrete([9, 9])

        self.points = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4),(1, 5),(1, 6),(1, 7),(1, 9),(1, 10)]
        self.adj = cu.coordinate2adjacent(self.points)
        self.step_cnt = 0
        self.total_reward = 0

        # obs[i] == qubit_nums 说明该位置为空，
        self.qubit_nums = 5
        self.flag = self.qubit_nums
        self.obs = [0,1,2,3,4,5,5,5,5]

        self.circuit = self.get_criruit()

        #circuit 相关变量
        #全连接
        self.adj = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1],
               [2, 5], [3, 0], [3, 4], [3, 6], [4, 1], [4, 3], [4, 5],
               [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 4],
               [7, 6], [7, 8], [8, 5], [8, 7]]
        self.qr =self.circuit.qubits
        layout = [None] * len(self.obs)
        for i in range(len(self.obs)):
            v = self.obs[i]
            if v != self.flag:
                layout[i] = self.qr[v]
        #默认score
        self.default_score = cu.get_circuit_score(self.circuit, self.adj, layout)

        #last record
        self.last_score = self.default_score
        self.last_action = np.array(0)

    def make_obs_space(self):
        space = MultiDiscrete(np.array([[6] * 9] ))

        return space
    def _get_info(self):
        return {'info':'this is info'}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.obs = [0,1,2,3,4,5,5,5,5]
        self.step_cnt = 0
        self.total_reward = 0

        layout = [None] * len(self.obs)
        for i in range(len(self.obs)):
            v = self.obs[i]
            if v != self.flag:
                layout[i] = self.qr[v]
        # 上个动作获取到的score
        self.default_score = cu.get_circuit_score(self.circuit, self.adj, layout)

        self.last_score = self.default_score
        self.last_action = np.array(0)

        info = self._get_info()
        return self.obs, info

    def step(self, action):
        self.step_cnt+=1
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward,observation = self._get_rewards(action)
        info = self._get_info()

        terminated = False
        truncated = False
        if self.total_reward <= -2:
            terminated = True
            #print('step_cnt = %r cut'%self.step_cnt)
        if self.total_reward == 1 or self.step_cnt==40:
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

    def get_criruit(self):
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.cx(0, 4)
        return  circuit

    #交互两个 points的位置，points 可以是空的
    def switch(self,position_1,position_2):

        temp = self.obs[position_2]
        # 位置 2 的值设置为1
        self.obs[position_2] = self.obs[position_1]
        # 源位置设置为空 (flag)
        self.obs[position_1] = temp
        return True

    def _get_rewards(self,action):
        #print(action)
        reward = -2
        if action[0] == action[1]:
            return -1.1, self.obs

        #防止出现相同的动作（原地摇摆）
        if  np.array_equal(action, self.last_action) \
            or np.array_equal(np.flip(action), self.last_action):

            return -1, self.obs

        self.last_action = action
        #交换位置
        self.switch(action[0],action[1])

        #计算score
        layout = [None] * len(self.obs)
        for i in range(len(self.obs)):
            v = self.obs[i]
            if v != self.flag:
                layout[i] = self.qr[v]
        # score 越低越好
        score = cu.get_circuit_score(self.circuit, self.adj, layout)

        if score is not None :
                #和上一次的比较
                if score >= self.last_score:
                    reward = -1.5*((score - self.last_score)/self.default_score)
                #和默认分数比较
                else:
                    reward = (self.default_score-score)/self.default_score

        else:
            reward = -2

        self.last_score = score

        self.total_reward*=0.95
        self.total_reward+=reward
        return reward,self.obs

    def _close_env(self):
        logger.info('_close_env')


if __name__ == '__main__':
    pass