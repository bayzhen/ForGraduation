import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Create the actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)
        return x


# Create the critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64 + action_size, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = self.fc1(state)
        x = torch.relu(x)
        x = torch.cat((x, action), dim=-1)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


device = "cuda:0"

# Create the environment
env = gym.make('LunarLander-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the actor and critic networks
actor = Actor(state_size, action_size)
critic = Critic(state_size, action_size)

# Set up the optimizers
# Set up the optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Train the networks
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        # Sample an action from the actor network
        action_probs = actor(torch.from_numpy(state).float())
        action = torch.multinomial(action_probs, 1).item()

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        # Compute the value of the current state-action pair
        state_action_value = critic(torch.from_numpy(state).float(), torch.from_numpy(np.array([action])).float())

        # Compute the value of the next state
        next_state_value = critic(torch.from_numpy(next_state).float(), action_probs)

        # Compute the temporal difference error
        td_error = reward + 0.99 * next_state_value - state_action_value

        # Update the critic network
        critic_optimizer.zero_grad()
        td_error.backward()
        critic_optimizer.step()

        # Update the actor network
        actor_optimizer.zero_grad()
        actor_loss = -td_error * action_probs
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the state
        state = next_state
