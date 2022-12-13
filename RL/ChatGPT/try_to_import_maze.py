import gym
import numpy as np
from typing import Any, Dict, Generator, List, Optional, Union
import matplotlib.pyplot as plt
import torch.cuda
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from scipy.stats import norm
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples
)
from gym import spaces
import torch as th
from env_simulator import EnvSimulator
from maze import MazeBigSimple


# Create the CartPole environment using the make_vec_env function from stable-baselines
game_env = MazeBigSimple()

# Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
model = DQN("MlpPolicy",
            env=game_env,
            verbose=1,
            tensorboard_log='./DQN/')

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000000)
