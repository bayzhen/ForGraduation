import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.dqn.policies import CnnPolicy

# Create the CartPole environment using the make_vec_env function from stable-baselines
env = make_vec_env('CartPole-v1', n_envs=1)


# Define the replay buffer class
class PrioritizedReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, experience):
        # Add the experience to the buffer and remove the oldest experience if the buffer is full
        self.buffer.append(experience)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer with replacement, using the priorities as the probability distribution
        priorities = [e[2] for e in self.buffer]
        batch = np.random.choice(self.buffer, size=batch_size, p=priorities)
        return batch

    def update_priorities(self, indices, priorities):
        # Update the priorities of the experiences at the given indices
        for i, priority in zip(indices, priorities):
            self.buffer[i] = (self.buffer[i][0], self.buffer[i][1], priority)


# Define the callback class
class PrioritizedExperienceReplayCallback(BaseCallback):
    def __init__(self, replay_buffer, batch_size):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def _on_step(self):
        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Calculate the priorities for each experience in the batch
        priorities = []
        for exp in batch:
            # Get the predicted action and true action for the current experience
            pred_act, true_act = exp[1][0], exp[2]

            # Replace the predicted action with the true action and recalculate the best predicted action
            pred_act[true_act] = -np.inf
            new_pred_act = np.argmax(pred_act)

            # Set the priority to 1 if the best predicted action changed, otherwise set it to 0
            priorities.append(1 if new_pred_act != true_act else 0)

        # Update the priorities in the replay buffer
        self.replay_buffer.update_priorities(batch, priorities)


# Create the replay buffer and callback
replay_buffer = PrioritizedReplayBuffer(size=1000)
callback = PrioritizedExperienceReplayCallback(replay_buffer, batch_size=32)

# Create the DQN model using a CNN policy
model = DQN(CnnPolicy, env, replay_buffer_class=PrioritizedReplayBuffer, replay_buffer_kwargs={'size': 1000}, verbose=1)

# Train the model for 1000 steps, using the callback to update the priorities in the replay buffer
model.learn(total_timesteps=1000, callback=callback)

# Plot the distribution of priorities in the replay buffer
priorities = [e[2] for e in replay_buffer.buffer]
plt.hist(priorities, bins=2)
plt.show()
