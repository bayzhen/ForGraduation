import gym
from override.ODQN import DQN
import numpy as np
import os


class Labrary:

    def __init__(self):
        pass

    env_name = "MazeBigSimple-v0"
    env = gym.make(env_name)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='./DQN/')

    files = next(os.walk(model.algorithm_name))[1]
    index_list = []
    for file in files:
        index_list.append(int(file.split('_')[1]))

    max_index = max(index_list)
    last_file = files[0].split('_')[0] + '_' + str(max_index)
    latest_path = model.algorithm_name + '/' + last_file
    path_to_save_model = latest_path + '/' + env_name + "_model"

    model.load(path_to_save_model)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        print(action, reward)
        env.render()
        if done:
            print('done')
            obs = env.reset()


if __name__ == '__main__':
    library = Labrary()
