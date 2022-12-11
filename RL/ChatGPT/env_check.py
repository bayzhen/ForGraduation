from stable_baselines3.common.env_checker import check_env
import gym
env = gym.make('MazeBigSimple-v0')
check_env(env)
