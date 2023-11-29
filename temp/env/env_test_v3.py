import math

import gymnasium as gym
import numpy
import numpy as np
import copy

import qiskit
from gymnasium import spaces
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from loguru import logger
import warnings

from utils.graph_util import GraphUtil as gu
from utils.points_util import PointsUtil as pu
from config import get_args
simulator = AerSimulator()
'''
不给定硬件拓扑，让智能体自己寻找最佳连接
'''
from utils.circuit_util import CircutUtil as cu
warnings.filterwarnings("ignore")
class CircuitEnvTest_v3(gym.Env):
    def __init__(self, render_mode=None,**kwargs):
        args = get_args()
        self.debug = kwargs.get('debug')

        # obs[i] == qubit_nums 说明该位置为空，
        # circuit 相关变量
        self.circuit = self.get_criruit(args.circuit_name)
        self.qubit_nums = len(self.circuit.qubits)
        # self.qr =self.circuit.qubits

        self.observation_space = spaces.Box(0,1,(self.qubit_nums, self.qubit_nums),dtype=np.uint8,)
        self.action_space = MultiDiscrete([self.qubit_nums, self.qubit_nums, 2, 2])

        self.max_step = 30
        self.max_edges=4
        self.stop_thresh = -5



    def _get_info(self):
        return {'info':'this is info'}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.graph = gu.get_new_graph(self.qubit_nums)
        self.adj = gu.get_adj_list(self.graph)
        self.obs = gu.get_adj_matrix(self.graph)

        self.step_cnt = 0
        self.total_reward = 0


        # 上个动作获取到的score
        self.default_score = cu.get_circuit_score1(self.circuit, self.adj)
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
        if self.total_reward <= self.stop_thresh or self.step_cnt==self.max_step :
            terminated = True

        if action[3] == 1:
            if self.debug: print('early stop at %r total reward = %r'% ( self.step_cnt,self.total_reward))
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


    def get_criruit(self,name:str):
        # circuit = QuantumCircuit(5)
        # circuit.cx(0, 1)
        # circuit.cx(0, 2)
        # circuit.cx(0, 3)
        # circuit.cx(0, 4)
        circuit = cu.get_from_qasm(name)
        return circuit

    def _get_rewards(self,act):
        action = np.array([0, 0])
        action[0] = act[0]
        action[1] = act[1]

        opt = act[2]
        #print(action)
        reward = self.stop_thresh
        if action[0] == action[1]:
            return self.stop_thresh/5, self._get_obs()

        #防止出现相同的动作（原地摇摆）
        # if  np.array_equal(action, self.last_action) \
        #     or np.array_equal(np.flip(action), self.last_action):
        #
        #     return self.stop_thresh/5, self._get_obs()

        score = None
        #执行动作
        if opt==1:
            # 执行删除边的操作
            if self.graph.has_edge(action[0], action[1]):
                self.graph.remove_edge(action[0], action[1])
                self.adj = gu.get_adj_list(self.graph)
                score = cu.get_circuit_score1(self.circuit, self.adj)
            else:
                reward = self.stop_thresh
        else:
            #超出最大连通分量
            if len(self.graph.edges(action[0]))== self.max_edges or \
                    len(self.graph.edges(action[1]))== self.max_edges:
                reward = self.stop_thresh
            else:
                # 执行增加边的操作
                self.graph.add_edge(action[0],action[1])
                self.adj = gu.get_adj_list(self.graph)
                score = cu.get_circuit_score1(self.circuit, self.adj)


        if score is not None :
            k1 = (self.default_score - score)/self.default_score
            k2 = (self.last_score - score)/self.last_score
            # 和上一次的比较
            # if score >= self.best_score:
            #     reward = 0.5*((self.best_score-score)/self.default_score)-0.02
            #
            # #和默认分数比较
            # else:
            #     reward = 2*(self.default_score-score)/self.default_score
            #     self.best_score = score

            if k1 > 0:
                reward = (math.pow((1 + k1), 2) - 1) * math.fabs(1 + k1)
            else:
                reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs(1 - k2)

        else:
            reward = self.stop_thresh

        self.last_score = score
        #每多走一步惩罚一次
        #reward = reward-(0.01 * self.step_cnt)
        self.total_reward*=0.99
        self.total_reward+=reward
        self.last_action = action

        if self.debug:
            print('step%r obs=%r, score=%r reward=%r'%(self.step_cnt,self.obs,score,reward))

        self.obs = gu.get_adj_matrix(self.graph)

        return reward,self._get_obs()

    def _close_env(self):
        logger.info('_close_env')


if __name__ == '__main__':
    print(spaces.Box(
        0,
        1,
        (5, 5),
        dtype=np.uint8,
    ).sample())