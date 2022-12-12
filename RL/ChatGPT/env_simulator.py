import torch as th
import torch.nn as nn


class EnvSimulator(nn.Module):
    def __init__(self, observation_space, action_space):
        super(EnvSimulator, self).__init__()
        self.l1 = nn.Linear(observation_space + action_space, 16)
        self.l2 = nn.Linear(16, 1 + observation_space)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        return self.l2(x)
