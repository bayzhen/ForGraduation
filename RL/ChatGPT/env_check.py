from stable_baselines3.common.env_checker import check_env
from RL.maze import Maze

env = Maze(use_image=False)
check_env(env)
