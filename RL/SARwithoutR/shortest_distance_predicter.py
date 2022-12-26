import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x


class SDP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.model = Network(input_size, hidden_size1, hidden_size2, output_size)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    def train(self, state, goal_state, action, n):
        input_data = np.concatenate((state, goal_state))
        input_data = torch.tensor(input_data).to(torch.float32)

        # Forward pass
        output = self.model(input_data)

        # Compute the loss
        loss = self.loss_fn(output, n)

        # Backward pass
        loss.backward()

        # Optimize the model parameters
        self.optimizer.step()

    def predict(self, state, goal_state):
        input_data = np.concatenate((state, goal_state))
        input_data = torch.tensor(input_data).to(torch.float32)
        # Forward pass
        output = self.model(input_data)
        return output
