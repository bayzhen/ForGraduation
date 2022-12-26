import torch

from shortest_distance_predicter import SDP
from RL.maze import Maze
import gym
import numpy as np



class SA:
    def __init__(self, env: gym.Env, goal_state):
        self.env = env
        self.goal_state = goal_state
        state_size = env.observation_space.shape[0]
        action_size = 1
        distance_size = 1
        self.SDP = SDP(state_size * 2, 32, 32, distance_size)
        self.replay_buffer = ReplayBuffer(1000,model=self.SDP)
        self.epsilon = 0.7

    def train(self, trajectory: list):
        for i in range(len(trajectory) - 1):
            data = trajectory[i]
            state = data[0]
            action = data[1]
            next_state = data[i + 1][0]
            n = 0
            self.SDP.train(state, next_state, action, n)

    def learn(self, total_timesteps):
        state = self.env.reset()
        for _ in range(total_timesteps):
            steps = self.SDP.predict(state, self.goal_state)
            best_action = np.argmax(steps, axis=0)
            action = np.random.choice([best_action, 0, 1], size=1,
                                      p=[self.epsilon, (1 - self.epsilon) / 2, (1 - self.epsilon) / 2])
            action = int(action[0])
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.add(state, action, reward, done, next_state)
            state = next_state
            if done:
                action = -1
                trajectory = []
                state = self.env.reset()
        print(1)


if __name__ == '__main__':
    game_env = Maze(use_image=False, is_random=False)
    game_env = gym.make('CartPole-v0')
    goal_state = np.array([0, 0, 0, 0], float)
    # Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
    model = SA(env=game_env, goal_state=goal_state)

    # Train the model for 1000 steps, using the replay buffer to store experiences
    model.learn(total_timesteps=1000)
