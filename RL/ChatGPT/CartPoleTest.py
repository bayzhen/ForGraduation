import random

import gym

# Create a new CartPole environment
env = gym.make("CartPole-v1")

# Define the number of episodes to train the agent
num_episodes = 1000

# Train the agent for the specified number of episodes
for episode in range(num_episodes):
  # Reset the environment at the start of each episode
  state = env.reset()

  # Run the episode until it is done
  done = False
  while not done:
    action = random.choice([0,1])
    # Take the action and observe the next state and reward
    next_state, reward, done, infos = env.step(action)

    print(done,infos)


  # Print the final reward for the episode
  print(f"Episode {episode}: reward {reward}")
