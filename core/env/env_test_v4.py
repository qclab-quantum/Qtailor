import math

import gymnasium as gym
import numpy
import numpy as np
import copy

import qiskit
from gymnasium import spaces, register
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from gymnasium.spaces.utils import flatten_space
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from loguru import logger
import warnings

from utils.concurrent_set import ConcurrentMap
from utils.graph_util import GraphUtil as gu, GraphUtil
from utils.points_util import PointsUtil as pu
from config import get_args, ConfigSingleton
import os
os.environ["SHARED_MEMORY_USE_LOCK"] = '1'

from shared_memory_dict import SharedMemoryDict
simulator = AerSimulator()
'''
不给定硬件拓扑，让智能体自己寻找最佳连接
v4 更新： 减小 obs 的体积，删去对称的冗余信息，提升模型效率
'''
from utils.circuit_util import CircuitUtil as cu
warnings.filterwarnings("ignore")
class CircuitEnvTest_v4(gym.Env):
    def __init__(self, render_mode=None,kwargs = {'debug':False},env_config=None):
        args = ConfigSingleton().get_config()
        self.debug = kwargs.get('debug')

        # obs[i] == qubit_nums 说明该位置为空，
        # circuit 相关变量
        smd = SharedMemoryDict(name='tokens',size=1024)
        qasm = smd['qasm']
        self.circuit = self.get_criruit(qasm)

        self.qubit_nums = len(self.circuit.qubits)
        # self.qr =self.circuit.qubits

        obs_size = int((self.qubit_nums * self.qubit_nums - self.qubit_nums ) / 2)
        self.observation_space = flatten_space(spaces.Box(0,1,(1,obs_size),dtype=np.uint8,))
        self.action_space = MultiDiscrete([self.qubit_nums, self.qubit_nums, 2, 2])
        #self.action_space = MultiDiscrete([self.qubit_nums, self.qubit_nums, 2])

        self.max_step = 15
        self.max_edges=4
        self.stop_thresh = -2

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
        self.default_score = cu.get_circuit_score(self.circuit, self.adj)
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
        if self.total_reward <= self.stop_thresh \
                or reward == self.stop_thresh \
                or self.step_cnt==self.max_step :
            terminated = True

        # if action[3] == 1:
        #     if self.debug: print('early stop at %r total reward = %r'% ( self.step_cnt,self.total_reward))
        #     terminated = True

        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')

    def close(self):
        self._close_env()


    def _get_obs(self):
        #flattened_matrix = np.array(copy.deepcopy(self.obs)).flatten()
        return GraphUtil.lower_triangle_to_1d_array(self.obs)

    def _get_info(self):
        return {"info":"this is info"}


    def get_criruit(self,name:str):
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
            return self.stop_thresh/10, self._get_obs()

        #防止出现相同的动作（原地摇摆）
        if  np.array_equal(action, self.last_action) \
            or np.array_equal(np.flip(action), self.last_action):

            return self.stop_thresh/5, self._get_obs()

        score = None
        #执行动作
        if opt==1:
            # 执行删除边的操作
            if self.graph.has_edge(action[0], action[1]):
                self.graph.remove_edge(action[0], action[1])
                self.adj = gu.get_adj_list(self.graph)
                score = cu.get_circuit_score(self.circuit, self.adj)
            else:
                #reward = self.stop_thresh
                #要删除的边不存在，无法执行操作
                return self.stop_thresh / 10, self._get_obs()
        else:
            #超出最大连通分量，无法执行操作
            if len(self.graph.edges(action[0]))== self.max_edges or \
                    len(self.graph.edges(action[1]))== self.max_edges:
                #reward = self.stop_thresh
                return self.stop_thresh , self._get_obs()
            else:
                # 执行增加边的操作
                self.graph.add_edge(action[0],action[1])
                self.adj = gu.get_adj_list(self.graph)
                score = cu.get_circuit_score(self.circuit, self.adj)


        if score is not None :
            k1 = (self.default_score - score)/self.default_score
            k2 = (self.last_score - score)/self.last_score

            if k2 > 0:
                reward =    (math.pow((1 + k2), 2)-1)*(1 + k1)
            elif k2 < 0:
                reward = -1*(math.pow((1 - k2), 2)-1)*(1 - k1)
            else:
                reward = 0
            self.last_score = score
        else:
            reward = self.stop_thresh


        #每多走一步惩罚一次
        #reward = reward-(0.01 * self.step_cnt)
        self.total_reward*=0.99
        self.total_reward+=reward

        self.last_action = action

        self.obs = gu.get_adj_matrix(self.graph)
        if self.debug:
            print('action = %r,  step=%r , score=%r ,reward=%r  \n obs=%r,'%(action,self.step_cnt,score,reward,self.obs))

        return reward,self._get_obs()

    def _close_env(self):
        logger.info('_close_env')

    #取矩阵的左下三角


if __name__ == '__main__':
    register(
        id='CircuitEnvTest-v4',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='temp.env.env_test_v4:CircuitEnvTest_v4',
        max_episode_steps=4000000,
    )

    # Create the env to do inference in.
    env = gym.make("CircuitEnvTest-v4")
    obs, info = env.reset()
    print(obs)
    obs, info = env.reset()
    print(obs)
    #print(flatten_space(spaces.Box(0, 1, (1,10), dtype=np.uint8 )).sample())



