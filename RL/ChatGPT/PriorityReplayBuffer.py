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


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(PriorityReplayBuffer, self).__init__(buffer_size,
                                                   observation_space,
                                                   action_space,
                                                   device,
                                                   n_envs,
                                                   optimize_memory_usage,
                                                   handle_timeout_termination)
        if type(observation_space) is gym.spaces.box.Box:
            v0 = 1
            for i in observation_space.shape:
                v0 *= i
            state_input_size = v0
        if type(action_space) is gym.spaces.discrete.Discrete:
            action_input_size = 1
        self.env_simulator = EnvSimulator(state_input_size, action_input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.env_simulator.parameters(), lr=0.001)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        rewards = np.expand_dims(self.rewards, axis=2)
        if self.full:
            nn_input = np.concatenate((self.observations, self.actions), axis=2)
            nn_output = np.concatenate((rewards, self.next_observations), axis=2)
        else:
            nn_input = np.concatenate((self.observations, self.actions), axis=2)[:self.pos]
            nn_output = np.concatenate((rewards, self.next_observations), axis=2)[:self.pos]
        nn_input = nn_input.astype(np.float32)
        nn_output = nn_output.astype(np.float32)
        nn_input = torch.tensor(nn_input).to(self.device)
        nn_output = torch.tensor(nn_output).to(self.device)
        criterion = torch.nn.MSELoss()
        loss = criterion(self.env_simulator(nn_input), nn_output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        print(f"loss: {loss:>7f}")
        return super(PriorityReplayBuffer, self).sample(batch_size, env)


# Create the CartPole environment using the make_vec_env function from stable-baselines
game_env = make_vec_env('CartPole-v1', n_envs=1)

# Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
model = DQN("MlpPolicy",
            env=game_env,
            replay_buffer_class=PriorityReplayBuffer,
            verbose=1,
            tensorboard_log='./DQN/')

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000000)
