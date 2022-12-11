import numpy as np
from typing import Any, Dict, Generator, List, Optional, Union
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from scipy.stats import norm
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples
)


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, size, observation_space, action_space, device, n_envs, optimize_memory_usage):
        # Call the __init__() method of the parent class
        super(PriorityReplayBuffer, self).__init__(size,
                                                   observation_space=observation_space,
                                                   action_space=action_space,
                                                   device=device,
                                                   n_envs=n_envs,
                                                   optimize_memory_usage=optimize_memory_usage,
                                                   )
        self.trajectory = []

    def add(self, obs, next_obs, action, reward, done, infos):
        # Call the add() method of the parent class
        self.trajectory.append([obs, next_obs, action, reward, done, infos])
        if done.item() is True:
            if len(self.trajectory) > 100:
                train_list = self.trajectory[int(len(self.trajectory) / 2):]
            else:
                train_list = self.trajectory
            for step in train_list:
                obs = step[0]
                next_obs = step[1]
                action = step[2]
                reward = step[3]
                done = step[4]
                infos = step[5]
                super(PriorityReplayBuffer, self).add(obs, next_obs, action, reward, done, infos)
            self.trajectory.clear()

    def my_sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        # Calculate the mean and standard deviation of the rewards
        super(PriorityReplayBuffer, self).sample()
        rewards = self.rewards
        mean = np.mean(rewards)
        std = np.std(rewards)

        x = np.linspace(-5, 5, 100)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y)
        plt.show()

        # Calculate the probability of each reward using the normal distribution
        probabilities = norm.pdf(rewards, mean, std)
        probabilities = np.array(probabilities).flatten()
        probabilities = 1 / probabilities
        probabilities = probabilities / sum(probabilities)
        indexes = np.random.choice(len(rewards), p=probabilities, size=(batch_size))
        return self._get_samples(indexes, env=env)


# Create the CartPole environment using the make_vec_env function from stable-baselines
game_env = make_vec_env('LunarLander-v2', n_envs=1)

# Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
model = DQN("MlpPolicy",
            env=game_env,
            verbose=1,
            tensorboard_log='./DQN/')

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000000)
