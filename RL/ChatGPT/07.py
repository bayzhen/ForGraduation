import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
# Import the MlpPolicy class
from stable_baselines3.dqn.policies import MlpPolicy


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, observation_space, action_space, buffer_size):
        super().__init__(observation_space, action_space, buffer_size)
        self.priorities = []

    def add(self, observation, action, reward, next_observation, done, priority):
        # Call the parent class add method to add the experience
        super().add(observation, action, reward, next_observation, done)
        # Store the priority of the experience in the 'priorities' list
        self.priorities.append(priority)

    def sample(self, batch_size):
        # Calculate the sum of all priorities in the buffer
        priority_sum = sum(self.priorities)
        # Generate a list of probabilities for each experience in the buffer
        probabilities = [p / priority_sum for p in self.priorities]
        # Sample 'batch_size' experiences from the buffer using the generated probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        # Return the sampled experiences and their corresponding indices
        return experiences, indices


# Create the CartPole environment using the make_vec_env function from stable-baselines
env = make_vec_env('CartPole-v1', n_envs=1)

# Create the DQN model using an MlpPolicy and the PriorityReplayBuffer class
model = DQN(MlpPolicy, env, replay_buffer_class=PriorityReplayBuffer, verbose=1)

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000)

# Plot the distribution of priorities in the replay buffer
plt.hist(model.replay_buffer.priorities, bins=2)
plt.show()

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000)

# Plot the distribution of priorities in the replay buffer
plt.hist(model.replay_buffer.priorities, bins=2)
plt.show()
