
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG1DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        # adaptive pooling (works with any ECG length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x