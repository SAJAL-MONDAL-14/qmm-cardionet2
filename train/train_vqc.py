# import torch
# import numpy as np
# import pandas as pd
# import wfdb
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, classification_report

# from torch.utils.data import TensorDataset, DataLoader

# from models.quantum.vqc_model import VQCModel
# from dataset.feature_extractor import extract_ecg_features


# # ===============================
# # Device
# # ===============================
# device = "cuda" if torch.cuda.is_available() else "cpu"


# # ===============================
# # Load PTB-XL metadata
# # ===============================
# ptbxl = pd.read_csv("data/ptbxl/ptbxl_database.csv")

# X = []
# y = []

# print("Preparing dataset...")

# for i in range(len(ptbxl)):

#     age = ptbxl.iloc[i]["age"]
#     sex = ptbxl.iloc[i]["sex"]

#     if pd.isna(age) or pd.isna(sex):
#         continue

#     path = os.path.join("data/ptbxl", ptbxl.iloc[i]["filename_hr"])

#     try:
#         signal, _ = wfdb.rdsamp(path)
#     except:
#         continue

#     ecg_features = extract_ecg_features(signal[:,0])

#     features = np.concatenate([[age, sex], ecg_features])

#     scp = ptbxl.iloc[i]["scp_codes"]

#     label = 0 if "NORM" in scp else 1

#     X.append(features)
#     y.append(label)


# X = np.array(X)
# y = np.array(y)

# print("Dataset shape:", X.shape)
# print("Class distribution:", np.bincount(y))


# # ===============================
# # Train/Test Split
# # ===============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


# # ===============================
# # Feature Scaling
# # ===============================
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # ===============================
# # Convert to PyTorch
# # ===============================
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


# # ===============================
# # DataLoader
# # ===============================
# train_dataset = TensorDataset(X_train, y_train)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=256,
#     shuffle=True
# )


# # ===============================
# # Model
# # ===============================
# model = VQCModel(X_train.shape[1]).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=20,
#     gamma=0.7
# )

# pos_weight = torch.tensor([9514/12285]).to(device)

# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# # ===============================
# # Training
# # ===============================
# print("\nTraining VQC...\n")

# for epoch in range(80):

#     model.train()

#     total_loss = 0

#     for xb, yb in train_loader:

#         xb = xb.to(device)
#         yb = yb.to(device)

#         optimizer.zero_grad()

#         out = model(xb)

#         loss = criterion(out, yb)

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")

#     scheduler.step()

# # ===============================
# # Evaluation
# # ===============================
# print("\nEvaluating Model...\n")

# model.eval()

# with torch.no_grad():
#     outputs = model(X_test)
#     probs = torch.sigmoid(outputs).cpu().numpy()

# labels = y_test.cpu().numpy()

# best_acc = 0
# best_t = 0.5

# for t in np.arange(0.3, 0.7, 0.01):
#     preds = (probs > t).astype(int)
#     acc = (preds == labels).mean()

#     if acc > best_acc:
#         best_acc = acc
#         best_t = t

# print("Best Threshold:", best_t)
# print("Best Accuracy:", best_acc)

# preds = (probs > best_t).astype(int)

# print("\nConfusion Matrix:")
# print(confusion_matrix(labels, preds))

# print("\nClassification Report:")
# print(classification_report(labels, preds))


"""
train_vqc.py  —  Complete Final Version
========================================
Just run:  python train/train_vqc.py
Model saved → saved_models/vqc_model.pth

CPU NOTE: Quantum simulation is slow on CPU.
MAX_SAMPLES=1000 is set for reasonable runtime (~2–4 hrs).
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics          import (
    accuracy_score, classification_report, confusion_matrix
)
from torch.utils.data          import TensorDataset, DataLoader
from models.quantum.vqc_model  import VQCModel
from dataset.dataset_builder   import build_flat_dataset


# ── Dataset (limited for CPU quantum simulation) ──────────
X, y = build_flat_dataset(multilead=True, max_samples=1000)

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Tensors ───────────────────────────────────────────────
device  = "cpu"

X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
y_tr_t  = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_te_t  = torch.tensor(X_test,  dtype=torch.float32)
y_te_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

loader  = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=32, shuffle=True
)

# ── Model ─────────────────────────────────────────────────
model  = VQCModel(X_tr_t.shape[1])
opt    = torch.optim.Adam(model.parameters(), lr=0.001)
sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.7)

n_neg  = (y_train == 0).sum()
n_pos  = (y_train == 1).sum()
pos_w  = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
crit   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

# ── Train ─────────────────────────────────────────────────
print("\nTraining VQC …  (this will take 2-4 hrs on CPU, be patient)")
EPOCHS = 40   # reduced from 80 for CPU

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    sched.step()
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={total/len(loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_te_t)).numpy().flatten()

labels = y_test

# find best threshold
best_acc, best_t = 0, 0.5
for t in np.arange(0.3, 0.7, 0.01):
    preds = (probs > t).astype(int)
    acc   = (preds == labels).mean()
    if acc > best_acc:
        best_acc, best_t = acc, t

preds = (probs > best_t).astype(int)
print(f"\nBest Threshold : {best_t:.2f}")
print("VQC Accuracy   :", accuracy_score(labels, preds))
print("\nConfusion Matrix:\n",      confusion_matrix(labels, preds))
print("\nClassification Report:\n", classification_report(labels, preds))

# ── Save ──────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/vqc_model.pth")
print("\n✓  Saved → saved_models/vqc_model.pth")