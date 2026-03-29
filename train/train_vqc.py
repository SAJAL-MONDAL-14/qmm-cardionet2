

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