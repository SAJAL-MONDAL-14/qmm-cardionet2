# import torch
# import numpy as np
# import pandas as pd
# import wfdb
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# from torch.utils.data import TensorDataset, DataLoader

# from models.classical.ann_model import ANNModel
# from dataset.feature_extractor import extract_ecg_features


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

#     ecg_features = extract_ecg_features(signal[:, 0])

#     features = np.concatenate([[age, sex], ecg_features])

#     scp = ptbxl.iloc[i]["scp_codes"]

#     if "NORM" in scp:
#         label = 0
#     else:
#         label = 1

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
# device = "cuda" if torch.cuda.is_available() else "cpu"

# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


# # ===============================
# # DataLoader (mini-batch training)
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
# model = ANNModel(X_train.shape[1]).to(device)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.0005
# )

# criterion = torch.nn.BCEWithLogitsLoss()


# # ===============================
# # Training
# # ===============================
# print("\nTraining ANN...\n")

# for epoch in range(50):

#     model.train()

#     total_loss = 0

#     for xb, yb in train_loader:

#         xb = xb.to(device)
#         yb = yb.to(device)

#         optimizer.zero_grad()

#         out = model(xb)

#         loss = criterion(out, yb)

#         loss.backward()

#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")


# # ===============================
# # Evaluation
# # ===============================
# model.eval()

# with torch.no_grad():

#     preds = model(X_test)

#     preds = torch.sigmoid(preds)

#     preds = (preds > 0.5).float()

#     accuracy = (preds == y_test).float().mean()


# print("\nANN Accuracy:", accuracy.item())

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test.cpu(), preds.cpu()))

# print("\nClassification Report:")
# print(classification_report(y_test.cpu(), preds.cpu()))




"""
train_ann.py  —  Complete Final Version
========================================
Just run:  python train/train_ann.py
Model saved → saved_models/ann_model.pth
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics          import (
    accuracy_score, classification_report, confusion_matrix
)
from torch.utils.data            import TensorDataset, DataLoader
from models.classical.ann_model  import ANNModel
from dataset.dataset_builder     import build_flat_dataset


# ── Dataset ──────────────────────────────────────────────
X, y = build_flat_dataset(multilead=True)

# ── Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ─────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Tensors ───────────────────────────────────────────────
device  = "cpu"   # CPU only

X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
y_tr_t  = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_te_t  = torch.tensor(X_test,  dtype=torch.float32)
y_te_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

loader  = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=256, shuffle=True
)

# ── Model ─────────────────────────────────────────────────
model  = ANNModel(X_tr_t.shape[1])
opt    = torch.optim.Adam(model.parameters(), lr=0.0005)
sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)

# class weight for imbalance
n_neg  = (y_train == 0).sum()
n_pos  = (y_train == 1).sum()
pos_w  = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
crit   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

# ── Train ─────────────────────────────────────────────────
print("\nTraining ANN …")
EPOCHS = 50

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    sched.step()
    print(f"Epoch {epoch:02d}/{EPOCHS}  loss={total/len(loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_te_t)).numpy().flatten()
    preds = (probs > 0.5).astype(int)

print("\nANN Accuracy :", accuracy_score(y_test, preds))
print("\nConfusion Matrix:\n",      confusion_matrix(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ── Save ──────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/ann_model.pth")
print("\n✓  Saved → saved_models/ann_model.pth")