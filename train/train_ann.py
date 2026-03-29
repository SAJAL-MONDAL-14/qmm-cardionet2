

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