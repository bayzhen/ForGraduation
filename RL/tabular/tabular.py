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
        self.steps_list = np.zeros(shape=(10,))
        self.pos = 0
        self.rewards_list = np.zeros(shape=(10,))
        self.calculate_count = 0

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

    def display(self, step):
        print("--------------------")
        print("num_steps:", np.average(self.steps_list))
        print("sum_rewards:", np.average(self.rewards_list))
        print("steps:", step)
        print("calculate_count:", self.calculate_count)
        print('--------------------')
        if np.average(self.steps_list) < 20:
            return True

    def learn(self, total_steps):
        done_count = 0
        sum_rewards = 0
        step_count = 0
        step = 0
        for step in range(total_steps):
            if self.done:
                self.pre_state = self.env.reset()
                self.done = False
            else:
                pass
            action = self.get_action(self.pre_state)
            step_count += 1
            state, reward, done, info = self.env.step(action)
            max_next_value = self.get_value(state, self.get_best_action(state))
            pre_state_action_value = self.get_value(self.pre_state, action)
            value = pre_state_action_value + self.alpha * (max_next_value + reward - pre_state_action_value)
            self.calculate_count += 1
            self.set_value(self.pre_state, action, value)
            self.pre_state = state
            if done:
                done_count += 1
                self.done = done
                self.steps_list[self.pos] = step_count
                self.rewards_list[self.pos] = sum_rewards
                self.pos = (self.pos + 1) % 10
                end = False
                if done_count % 10 == 0:
                    end = self.display(step)
                sum_rewards = 0
                step_count = 0
                if end:
                    break
        return step, self.calculate_count


class UniqueQLearning(QLearning):
    def __init__(self, game_env):
        super(UniqueQLearning, self).__init__(game_env)
        self.q_table = np.zeros(shape=(11, 11, 4))

    def get_best_action(self, state):
        values = self.q_table[state[0], state[1]]
        max_value = np.max(values)
        indexes = []
        for i in range(4):
            if values[i] == max_value:
                indexes.append(i)
        return np.random.choice(indexes)

    def set_value(self, state: np.ndarray, action: int, value: float):
        self.q_table[state[0], state[1], action] = value

    def get_value(self, state: np.ndarray, action: int):
        return self.q_table[state[0], state[1], action]


class BetterQLearning(UniqueQLearning):

    def learn(self, total_steps):
        done_count = 0
        sum_rewards = 0
        step_count = 0
        for step in range(total_steps):
            if self.done:
                self.pre_state = self.env.reset()
                self.done = False
            else:
                pass
            action = self.get_action(self.pre_state)
            step_count += 1
            state, reward, done, info = self.env.step(action)

            # only for maze
            if reward == 1:
                for action in range(self.env.action_space.n):
                    self.set_value(state, action, 100)

            max_next_value = self.get_value(state, self.get_best_action(state))
            if max_next_value != 0:
                pre_state_action_value = self.get_value(self.pre_state, action)
                value = pre_state_action_value + self.alpha * (reward + max_next_value - pre_state_action_value)
                self.calculate_count += 1
                self.set_value(self.pre_state, action, value)

            self.pre_state = state
            if done:
                done_count += 1
                self.done = done
                self.steps_list[self.pos] = step_count
                self.rewards_list[self.pos] = sum_rewards
                self.pos = (self.pos + 1) % 10
                if done_count % 10 == 0:
                    self.display(step)
                sum_rewards = 0
                step_count = 0


def algorithms_compare(*models: QLearning):
    data = []
    game_env = Maze(use_image=False, is_random=False, x=10, y=10, hit_wall_reward=0, walk_reward=0)
    for model in models:
        for i in range(100):
            model.__init__(game_env)
            step, calculate_count = model.learn(1000000)

    pass


if __name__ == '__main__':
    env = Maze(use_image=False, is_random=False, x=10, y=10, hit_wall_reward=0, walk_reward=0)
    q_learning = BetterQLearning(env)
    q_learning.learn(1000000)
