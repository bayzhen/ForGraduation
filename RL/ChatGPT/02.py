import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# create the CartPole environment
env = gym.make('MazeBigSimple-v0')
# wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# create the reinforcement learning model
model = A2C('MlpPolicy', env, verbose=1)

# train the model for 1000 steps
model.learn(total_timesteps=100000)

# save the trained model
model.save("cartpole_model")

# use the trained model to make predictions
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
