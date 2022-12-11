import gym
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Create the stable-baselines3 model
model = stable_baselines3.DQN('MlpPolicy', env, verbose=1)


# Define the callback that will update the priorities in the replay buffer
class PrioritizationCallback(BaseCallback):
    def __init__(self):
        super(PrioritizationCallback, self).__init__()
        self.priorities = []

    def _on_step(self):
        # Get the current state and action from the 'locals' dictionary
        state = self.locals['new_obs']
        action = self.locals['actions']

        # Use the model to predict the best action in the current state
        action_values, _ = self.model.predict(state)
        predicted_action = action_values.argmax()
        if predicted_action == action:
            priority = 0
        else:
            priority = 1
        # Calculate the priority of the current experience
        self.priorities.append(priority)


# Train the model with the prioritization callback
model.learn(total_timesteps=1000, callback=PrioritizationCallback())

# Plot the distribution of priorities using matplotlib
plt.hist(PrioritizationCallback.priorities)
plt.show()
