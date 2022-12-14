from stable_baselines3 import PPO
from RL.maze import Maze

# Create the CartPole environment using the make_vec_env function from stable-baselines
game_env = Maze(use_image=False, is_random=False)

# Create the DQN model using a CNN policy and the PriorityReplayBuffer instance
model = PPO("MlpPolicy",
            env=game_env,
            verbose=2,
            tensorboard_log='./DQN/')

# Train the model for 1000 steps, using the replay buffer to store experiences
model.learn(total_timesteps=1000000)
