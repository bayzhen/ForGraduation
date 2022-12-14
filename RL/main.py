import gym
from stable_baselines3 import DQN
import numpy as np
import os


def get_last_path(algorithm):
    files = next(os.walk(algorithm))[1]
    index_list = []
    for file in files:
        index_list.append(int(file.split('_')[1]))
    max_index = max(index_list)
    last_file = files[0].split('_')[0] + '_' + str(max_index)
    latest_path = algorithm + '/' + last_file
    return latest_path


class TV:
    def __init__(self, env_name="MazeBigSimple-v0"):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.algorithm = None
        algorithm_list = {'DQN': self.dqn,
                          'PPO': self.ppo}
        self.algorithm = algorithm_list[input("请输入使用的算法: ")]
        self.model = None

    def train(self):
        self.algorithm()

    def ppo(self):
        pass

    def dqn(self):
        self.model = DQN("MlpPolicy", self.env, verbose=1, tensorboard_log='./tenborboard/')
        self.model.learn(total_timesteps=250000, log_interval=10)

        latest_path = get_last_path(self.algorithm)
        path_to_save_model = latest_path + '/' + self.env_name + "_model"
        path_to_save_replay_buffer = latest_path + "/replay_buffer"

        self.model.save(path_to_save_model)
        self.model.save_replay_buffer(path_to_save_replay_buffer)

        print("Finished")

    def test(self):
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                print('done')
                obs = self.env.reset()


if __name__ == '__main__':
    # train value
    tv = TV()
    tv.train()
    tv.test()
