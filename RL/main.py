import gym
from ODQN import DQN

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='./DQN/')
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_LunarLander-v2")
model.save_replay_buffer("DQN_lunar_lander_replay_buffer")

del model  # remove to demonstrate saving and loading

model = DQN.load("dqn_LunarLander-v2")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
