import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
import matplotlib.pyplot as plt

# Set up the environment
env = gym.make('CartPole-v1')

# Set up the stable-baseline3 agent with a discrete action space
model = stable_baselines3.A2C(env, verbose=1)


# Define a callback to track the distribution of priorities
class PriorityDistribution(BaseCallback):
    def __init__(self):
        self.priorities = []

    def on_episode_end(self, episode, logs):
        self.priorities.append(logs['priority'])

    def on_training_end(self, logs):
        plt.hist(self.priorities)
        plt.show()


# Define the replay buffer
class PriorityReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size
        self.priorities = []

    def add(self, experience, priority):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
            self.priorities.pop(0)

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        # Calculate the priority weights
        priority_sum = sum(self.priorities)
        priority_weights = [p / priority_sum for p in self.priorities]

        # Sample from the replay buffer using the priority weights
        indices = np.random.choice(len(self.buffer), size=batch_size, p=priority_weights)
        experiences = [self.buffer[i] for i in indices]
        return experiences

    def update_priorities(self, indices, new_priorities):
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p


# Create the replay buffer and callback
replay_buffer = PriorityReplayBuffer(size=10000)
callback = PriorityDistribution()

# Train the agent using the replay buffer and callback
model.learn(total_timesteps=10000, callback=callback, replay_buffer=replay_buffer)
