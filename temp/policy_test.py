import argparse
import os
import pprint
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, DDPGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from gymnasium import register