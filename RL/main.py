import gym
from override.ODQN import DQN
import numpy as np
import os

env_name = "LunarLander-v2"
env = gym.make(env_name)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='./DQN/')
model.learn(total_timesteps=10000, log_interval=4)

latest_path = model.algorithm_name + '/' + next(os.walk(model.algorithm_name))[1][-1]
path_to_save_model = latest_path + '/' + env_name + "_model"
path_to_save_replay_buffer = latest_path + "/replay_buffer"

model.save(path_to_save_model)
model.save_replay_buffer(path_to_save_replay_buffer)
del model  # remove to demonstrate saving and loading

# 是用replay_buffer，还是用deviation_of_prediction
# replay_buffer可以作为分析训练价值的数据
# deviation_of_prediction可以直接计算出当前的预测的偏差，并判断是否有训练的价值

print("Finished")
