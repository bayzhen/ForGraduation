from stable_baselines3.common.env_checker import check_env
from maze import Maze
import gym
env = Maze(use_image=False)
check_env(env)
