# import torch
# import torch.nn as nn

# from models.ecg_1dcnn import ECG1DCNN
# from models.quantum_layer import QuantumLayer


# class MultimodalQuantumNet(nn.Module):

#     def __init__(self):

#         super().__init__()

#         # ECG CNN
#         self.ecg_net = ECG1DCNN()

#         # Clinical branch (age, sex)
#         self.clinical_net = nn.Sequential(

#             nn.Linear(2, 64),
#             nn.ReLU(),

#             nn.Linear(64, 32),
#             nn.ReLU()
#         )

#         # Fusion layer
#         self.fusion = nn.Sequential(

#             nn.Linear(128 + 32, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(256, 64),
#             nn.ReLU(),

#             nn.Linear(64, 8)
#         )

#         # Quantum layer
#         self.quantum = QuantumLayer()

#         # Final classifier
#         self.classifier = nn.Sequential(

#             nn.Linear(8, 16),
#             nn.ReLU(),

#             nn.Linear(16, 1)
#         )

#     def forward(self, clinical_x, ecg_x):

#         ecg_feat = self.ecg_net(ecg_x)

#         clin_feat = self.clinical_net(clinical_x)

#         fused = torch.cat([ecg_feat, clin_feat], dim=1)

#         fused = self.fusion(fused)

#         q = self.quantum(fused)

#         out = self.classifier(q)

#         return out












import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ecg_1dcnn import ECG1DCNN
from models.quantum_layer import QuantumLayer


# ── Original model (kept as is) ───────────────────────────
class MultimodalQuantumNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.ecg_net = ECG1DCNN()
        self.clinical_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.quantum    = QuantumLayer()
        self.classifier = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, clinical_x, ecg_x):
        ecg_feat  = self.ecg_net(ecg_x)
        clin_feat = self.clinical_net(clinical_x)
        fused     = torch.cat([ecg_feat, clin_feat], dim=1)
        fused     = self.fusion(fused)
        q         = self.quantum(fused)
        return self.classifier(q)


# ── New model used in training ─────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.b = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.relu(x + self.b(x))


class SharedEncoder(nn.Module):
    def __init__(self, ecg_dim):
        super().__init__()
        in_dim = ecg_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.res1 = ResBlock(128, 0.2)
        self.res2 = ResBlock(128, 0.15)
        self.proj = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )

    def forward(self, ecg_f, clin_f):
        x = torch.cat([ecg_f, clin_f], dim=1)
        return self.proj(self.res2(self.res1(self.net(x))))


class DualPathQuantumNet(nn.Module):
    def __init__(self, ecg_dim):
        super().__init__()
        self.encoder          = SharedEncoder(ecg_dim)
        self.classical_path   = nn.Sequential(
            ResBlock(64, 0.2), ResBlock(64, 0.15)
        )
        self.quantum_compress = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32,  8), nn.Tanh(),
        )
        self.quantum    = QuantumLayer()
        self.classifier = nn.Sequential(
            nn.Linear(72, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32,  1),
        )

    def forward(self, ecg_f, clin_f):
        shared  = self.encoder(ecg_f, clin_f)
        cls_out = self.classical_path(shared)
        q_out   = self.quantum(self.quantum_compress(shared))
        return self.classifier(torch.cat([cls_out, q_out], dim=1))