import torch
import torch.nn as nn
from models.quantum_layer import QuantumLayer


class VQCModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.quantum = QuantumLayer()

        self.classifier = nn.Linear(8, 1)

    def forward(self, x):

        x = self.net(x)

        q = self.quantum(x)

        return self.classifier(q)