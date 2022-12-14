from RL.maze import Maze
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

plt.ion()


class QLearning:
    def __init__(self, game_env: gym.Env):
        self.env = game_env
        self.q_table = {}
        self.pre_state = None
        self.alpha = 0.1
        self.epsilon = 0.2
        self.done = True
        self.step_count = 0
        self.sum_rewards = 0

    def get_value(self, state: np.ndarray, action: int):
        if len(state.shape) > 1:
            raise Exception("不支持图像。")
        key = str(state)[1:-1] + str(action)
        if key in self.q_table.keys():
            return self.q_table[key]
        else:
            return 0

    def set_value(self, state: np.ndarray, action: int, value: float):
        if len(state.shape) > 1:
            raise Exception("不支持图像。")
        key = str(state)[1:-1] + str(action)
        self.q_table[key] = value

    def train(self, total_steps):
        pass

    def display_matrix(self):
        pass

    def get_best_action(self, state):
        actions = []
        values = []
        for action in range(self.env.action_space.n):
            actions.append(action)
            values.append(self.get_value(state, action))
        max_value = max(values)
        best_actions = []
        for i in range(len(values)):
            if values[i] == max_value:
                best_actions.append(i)
        return random.choice(best_actions)

    def get_action(self, state):
        best_action = self.get_best_action(state)
        random_action = random.choice([x for x in range(self.env.action_space.n)])
        return np.random.choice([random_action, best_action], p=[self.epsilon, 1 - self.epsilon])

    def display(self):
        print("--------------------")
        print("num_steps:", self.step_count)
        print("sum_rewards:", self.sum_rewards)
        print('--------------------')

    def learn(self, total_steps):
        done_count = 0
        for _ in range(total_steps):
            if self.done:
                self.pre_state = self.env.reset()
                self.done = False
            else:
                pass
            action = self.get_action(self.pre_state)
            self.step_count += 1
            state, reward, done, info = self.env.step(action)
            # test
            if self.env.current_x == self.env.destination_x and self.env.current_y == self.env.destination_y:
                print('yes')

            self.sum_rewards += reward
            max_next_value = self.get_value(state, self.get_best_action(state))
            value = reward + self.alpha * (max_next_value - self.get_value(self.pre_state, action))
            self.set_value(self.pre_state, action, value)
            self.pre_state = state
            if done:
                done_count += 1
                if done_count % 10 == 0:
                    self.display()
                self.done = done
                self.step_count = 0
                self.sum_rewards = 0


if __name__ == '__main__':
    env = Maze(use_image=False, is_random=False, x=10, y=10)
    q_learning = QLearning(env)
    q_learning.learn(1000000)
