import gym
import numpy as np
from stable_baselines3 import DQN, PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        self.logger.dump(self.num_timesteps)
        return True


env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='./DQN/')
model.learn(total_timesteps=10000, log_interval=4, callback=TensorboardCallback())
model.save("dqn_lunar_lander_v2")

del model  # remove to demonstrate saving and loading
print("finished")