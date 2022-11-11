import gym
from override.ODQN import DQN
import numpy as np

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='./DQN/')
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_LunarLander-v2")
model.save_replay_buffer("DQN_lunar_lander_replay_buffer")
np.save("rewards_record.txt", model.rewards_record)

del model  # remove to demonstrate saving and loading

print("Finished")
