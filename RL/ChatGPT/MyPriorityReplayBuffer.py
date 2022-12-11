import gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.buffers import PrioritizedReplayBuffer

# Define the environment and the model
env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, prioritized_replay=True)

# Create a list of callbacks to track the distribution of priorities
dist_callback = PriorityDistributionTracker()
callbacks = CallbackList([dist_callback])

# Train the model and track the distribution of priorities
model.learn(total_timesteps=1000, callback=callbacks)

# Print the distribution of priorities
dist = dist_callback.get_distribution()
print(dist)

# Plot the distribution using matplotlib
plt.plot(dist)
plt.show()
