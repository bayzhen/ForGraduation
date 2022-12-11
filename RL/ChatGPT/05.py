import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import matplotlib.pyplot as plt


# Define the replay buffer
class PriorityReplayBuffer(object):
    def __init__(self, size):
        # Initialize the replay buffer with the given size
        self.size = size
        self.buffer = []

    def add(self, experience):
        # Add an experience to the replay buffer
        # If the buffer is full, remove the oldest experience to make room for the new one
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences from the replay buffer
        # The experiences are chosen based on their priority, rather than randomly
        # The priority of an experience is determined by whether the best predicted action
        # is the same as the true value of the experience
        experiences = []
        for _ in range(batch_size):
            # Find the experience with the highest priority
            max_priority = 0
            max_priority_idx = 0
            for i, exp in enumerate(self.buffer):
                if exp[2] > max_priority:
                    max_priority = exp[2]
                    max_priority_idx = i
            # Add the experience with the highest priority to the batch
            experiences.append(self.buffer.pop(max_priority_idx))
        # Return the batch of experiences
        return experiences

    def plot_distribution(self):
        # Plot the distribution of priorities in the replay buffer using matplotlib
        priorities = [exp[2] for exp in self.buffer]
        plt.hist(priorities)
        plt.show()


# Define a callback for tracking the distribution of priorities in the replay buffer
class PriorityDistributionTracker(BaseCallback):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.n_calls = 0

    def _on_step(self):
        self.n_calls += 1


# Create the environment
env = gym.make('MazeBigSimple-v0')

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Create the replay buffer
replay_buffer = PriorityReplayBuffer(size=10000)

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Create the callback for tracking the distribution of priorities in the replay buffer
callback = PriorityDistributionTracker(replay_buffer)

# Train the model
model.learn(total_timesteps=10000, callback=callback)

# Use the trained model to make predictions
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
