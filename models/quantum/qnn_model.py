import torch
import torch.nn as nn
from models.quantum_layer import QuantumLayer


class QNNModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(64, 8)

        self.quantum = QuantumLayer()

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        q = self.quantum(x)

        out = self.fc3(q)

        return out