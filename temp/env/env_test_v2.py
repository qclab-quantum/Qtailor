import gymnasium as gym
import numpy
import numpy as np
import copy
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
simulator = AerSimulator()

from utils.circuit_util import CircutUtil as cu
warnings.filterwarnings("ignore")
class CircuitEnvTest_v2(gym.Env):
    def __init__(self, render_mode=None,debug = False):

        self.debug = debug

        self.observation_space = self.make_obs_space()
        self.action_space = MultiDiscrete([9, 9, 2])

        self.points  = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2) ,(2,0),(2,1),(2,2)]
        self.adj = cu.coordinate2adjacent(self.points)
        self.step_cnt = 0
        self.total_reward = 0

        self.max_step = 10

        # obs[i] == qubit_nums 说明该位置为空，
        self.qubit_nums = 5
        self.flag = self.qubit_nums
        self.obs = [0,1,2,3,4,5,5,5,5]
        self.circuit = self.get_criruit()

        #circuit 相关变量
        self.qr =self.circuit.qubits
        #默认score
        self.default_score = None
        self.best_score = None
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
        self.best_score = self.default_score
        self.last_action = np.array(0)

        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        self.step_cnt+=1

        #assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward,observation = self._get_rewards(action)
        info = self._get_info()

        terminated = False
        truncated = False
        if self.total_reward <= -4 or self.step_cnt==self.max_step :
            terminated = True

        if action[2] == 1:
            if self.debug: print('early stop at %r'% self.step_cnt)
            terminated = True

        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _get_obs(self):
        #obs = np.array(cu.adjacency2matrix(cu.coordinate2adjacent(self.points)))
        return copy.deepcopy(self.obs)

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

    def _get_rewards(self,act):
        action = np.array([0, 0])
        action[0] = act[0]
        action[1] = act[1]
        #print(action)
        reward = -2
        if action[0] == action[1]:
            return -1, self._get_obs()

        #防止出现相同的动作（原地摇摆）
        if  np.array_equal(action, self.last_action) \
            or np.array_equal(np.flip(action), self.last_action):

            return -1, self._get_obs()

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
                if score >= self.best_score:
                    reward = ((self.best_score-score)/self.default_score)-0.05
                #和默认分数比较
                else:

                    reward = (self.default_score-score)/self.default_score
                    self.best_score = score
        else:
            reward = -2


        #每多走一步惩罚一次
        reward = reward-(0.01 * self.step_cnt)
        self.total_reward*=0.9
        self.total_reward+=reward
        if self.debug:
            print('step%r obs=%r, score=%r reward=%r'%(self.step_cnt,self.obs,score,reward))
        return reward,self._get_obs()

    def _close_env(self):
        logger.info('_close_env')


if __name__ == '__main__':
    pass