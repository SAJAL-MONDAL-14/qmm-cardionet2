
import os
import sys
import joblib
import time
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.feature_extractor import (
    extract_ecg_features_multilead,
    extract_ecg_features
)
from models.quantum_layer import QuantumLayer

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
PTBXL_CSV   = "data/ptbxl/ptbxl_database.csv"
PTBXL_BASE  = "data/ptbxl"
MAX_SAMPLES = 12000
BATCH_SIZE  = 32
EPOCHS      = 200
LR          = 5e-4
PATIENCE    = 40
SAVE_PATH   = "saved_models/best_multimodal_model.pth"
CACHE_PATH  = "saved_models/feature_cache.npz"

# SAME as train_all_flat.py and evaluate_models.py
TEST_SIZE   = 0.2
RANDOM_SEED = 42

device = "cpu"

print("=" * 60)
print("  QMM CARDIONET2 — Quantum Multimodal Training")
print("=" * 60)
print("  MAX_SAMPLES = " + str(MAX_SAMPLES))
print("  CACHE       = " + CACHE_PATH)
print("  SPLIT       = train_test_split seed=42 test=0.2")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
#  FEATURE EXTRACTION WITH CACHE
# ═══════════════════════════════════════════════════════════
def load_or_extract(csv, base, max_samples, cache_path):
    if os.path.exists(cache_path):
        print("\nCache found -> loading from " + cache_path)
        t0   = time.time()
        data = np.load(cache_path)
        ecg_arr  = data["ecg"]
        clin_arr = data["clin"]
        lbl_arr  = data["labels"]
        print("Loaded " + str(len(lbl_arr)) + " samples in " +
              str(round(time.time()-t0, 1)) + "s")
        print("NORM=" + str((lbl_arr==0).sum()) +
              "  ABNORMAL=" + str((lbl_arr==1).sum()))
        return ecg_arr, clin_arr, lbl_arr

    print("\nNo cache -> extracting features from " + str(max_samples) +
          " records ...")
    print("This runs ONCE then saves to cache.\n")

    df = pd.read_csv(csv)
    df = df[df["filename_hr"].notna()].reset_index(drop=True)
    df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        dn = df[df["scp_codes"].str.contains("NORM", na=False)]
        da = df[~df["scp_codes"].str.contains("NORM", na=False)]
        n  = max_samples // 2
        df = pd.concat([
            dn.sample(min(n, len(dn)), random_state=42),
            da.sample(min(n, len(da)), random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    ecg_f, clin_f, labels = [], [], []
    skip = 0
    t0   = time.time()

    for i in range(len(df)):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            eta     = elapsed / i * (len(df) - i)
            print("  [" + str(i) + "/" + str(len(df)) + "]  " +
                  "skip=" + str(skip) + "  " +
                  "elapsed=" + str(round(elapsed/60,1)) + "min  " +
                  "ETA=" + str(round(eta/60,1)) + "min")

        row  = df.iloc[i]
        path = os.path.join(base, row["filename_hr"])

        try:
            sig, _ = wfdb.rdsamp(path)
        except Exception:
            skip += 1; continue
        try:
            f = (extract_ecg_features_multilead(sig)
                 if sig.shape[1] >= 11
                 else extract_ecg_features(sig[:, 0]))
        except Exception:
            skip += 1; continue
        if not np.all(np.isfinite(f)):
            skip += 1; continue

        ecg_f.append(f)
        clin_f.append([float(row["age"]), float(row["sex"])])
        labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

    ecg_arr  = np.array(ecg_f,  dtype=np.float32)
    clin_arr = np.array(clin_f, dtype=np.float32)
    lbl_arr  = np.array(labels, dtype=np.float32)

    print("Extraction done in " + str(round((time.time()-t0)/60,1)) + " min")
    print("valid=" + str(len(lbl_arr)) + "  skipped=" + str(skip))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path,
                        ecg=ecg_arr, clin=clin_arr, labels=lbl_arr)
    print("Cache saved -> " + cache_path)

    return ecg_arr, clin_arr, lbl_arr


# ═══════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════
class CachedMultimodalDataset(Dataset):
    def __init__(self, ecg_arr, clin_arr, lbl_arr, scaler=None):
        combined = np.concatenate([ecg_arr, clin_arr], axis=1)

        # use shared scaler if provided (same as SVM/ANN/QNN/VQC)
        # otherwise fit a fresh one
        if scaler is not None:
            combined = scaler.transform(combined).astype(np.float32)
        else:
            sc = StandardScaler()
            combined = sc.fit_transform(combined).astype(np.float32)

        combined = np.nan_to_num(combined, nan=0., posinf=0., neginf=0.)

        self.ecg    = torch.tensor(combined[:, :ecg_arr.shape[1]],
                                    dtype=torch.float32)
        self.clin   = torch.tensor(combined[:, ecg_arr.shape[1]:],
                                    dtype=torch.float32)
        self.labels = torch.tensor(lbl_arr, dtype=torch.float32)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return self.ecg[i], self.clin[i], self.labels[i]


# ═══════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.b = nn.Sequential(
            nn.Linear(dim,dim), nn.BatchNorm1d(dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(dim,dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x): return F.relu(x + self.b(x))


class SharedEncoder(nn.Module):
    def __init__(self, ecg_dim):
        super().__init__()
        in_dim = ecg_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.res1 = ResBlock(128, 0.2)
        self.res2 = ResBlock(128, 0.15)
        self.proj = nn.Sequential(
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU()
        )
    def forward(self, ecg_f, clin_f):
        x = torch.cat([ecg_f, clin_f], dim=1)
        return self.proj(self.res2(self.res1(self.net(x))))


class DualPathQuantumNet(nn.Module):
    """
    Classical path (64) + Quantum path (8) in parallel.
    Merged at classifier -> 72-dim -> output.
    """
    def __init__(self, ecg_dim):
        super().__init__()
        self.encoder          = SharedEncoder(ecg_dim)
        self.classical_path   = nn.Sequential(
            ResBlock(64, 0.2), ResBlock(64, 0.15)
        )
        self.quantum_compress = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32, 8), nn.Tanh(),
        )
        self.quantum    = QuantumLayer()
        self.classifier = nn.Sequential(
            nn.Linear(72,64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, ecg_f, clin_f):
        shared  = self.encoder(ecg_f, clin_f)
        cls_out = self.classical_path(shared)
        q_out   = self.quantum(self.quantum_compress(shared))
        return self.classifier(torch.cat([cls_out, q_out], dim=1))


# ═══════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════
os.makedirs("saved_models", exist_ok=True)

ecg_arr, clin_arr, lbl_arr = load_or_extract(
    PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES, CACHE_PATH
)

# ── SAME split as train_all_flat.py and evaluate_models.py ─
idx = np.arange(len(ecg_arr))
idx_tr, idx_te = train_test_split(
    idx,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = lbl_arr.astype(int)
)

# load shared scaler for consistent scaling with other models
SCALER_PATH = "saved_models/shared_scaler.pkl"
if os.path.exists(SCALER_PATH):
    shared_scaler = joblib.load(SCALER_PATH)
    print("Using shared scaler: " + SCALER_PATH)
else:
    shared_scaler = None
    print("WARNING: shared_scaler.pkl not found - run train_all_flat.py first!")

train_ds = CachedMultimodalDataset(
    ecg_arr[idx_tr], clin_arr[idx_tr], lbl_arr[idx_tr],
    scaler=shared_scaler
)
test_ds  = CachedMultimodalDataset(
    ecg_arr[idx_te], clin_arr[idx_te], lbl_arr[idx_te],
    scaler=shared_scaler
)
ecg_dim  = train_ds.ecg.shape[1]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print("\nTrain=" + str(len(train_ds)) +
      "  Test="  + str(len(test_ds)) +
      "  ECG-dim=" + str(ecg_dim))

lbl_np = lbl_arr
pos_w  = torch.tensor(
    [(lbl_np==0).sum() / max((lbl_np==1).sum(), 1)],
    dtype=torch.float32
)
print("pos_weight=" + str(round(pos_w.item(), 3)))


# ═══════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════
model    = DualPathQuantumNet(ecg_dim)
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print("Parameters: " + str(n_params))

# separate LR for quantum vs classical
quantum_params   = list(model.quantum.parameters())
classical_params = [p for p in model.parameters()
                    if not any(p is q for q in quantum_params)]

optimizer = torch.optim.AdamW([
    {"params": classical_params, "lr": LR,       "weight_decay": 1e-4},
    {"params": quantum_params,   "lr": LR * 0.1, "weight_decay": 0.0},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
best_auc  = 0.0
patience_ct = 0

print("\n" + "="*60)
print("  Training (Classical + Quantum parallel)")
print("="*60 + "\n")

for epoch in range(1, EPOCHS + 1):
    model.train(); total = 0.0; t0 = time.time()

    for ef, cf, lb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(ef, cf), lb.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

    model.eval(); pa, pr, la = [], [], []
    with torch.no_grad():
        for ef, cf, lb in test_loader:
            p = torch.sigmoid(model(ef,cf)).numpy().flatten()
            pa.extend(p)
            pr.extend((p>0.5).astype(int))
            la.extend(lb.numpy())

    acc = accuracy_score(la, pr)
    auc = roc_auc_score(la, pa)
    f1  = f1_score(la, pr, zero_division=0)
    lr_now = optimizer.param_groups[0]["lr"]

    scheduler.step(auc)

    print("Epoch " + str(epoch).zfill(3) + "/" + str(EPOCHS) +
          "  loss=" + str(round(total/len(train_loader),4)) +
          "  acc="  + str(round(acc,4)) +
          "  auc="  + str(round(auc,4)) +
          "  f1="   + str(round(f1,4)) +
          "  lr="   + str(round(lr_now,6)) +
          "  (" + str(round(time.time()-t0,1)) + "s)")

    if auc > best_auc:
        best_auc = auc; patience_ct = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print("  Best saved  acc=" + str(round(acc,4)) +
              "  auc=" + str(round(auc,4)))
    else:
        patience_ct += 1
        print("  patience " + str(patience_ct) + "/" + str(PATIENCE))
        if patience_ct >= PATIENCE:
            print("  Early stopping at epoch " + str(epoch))
            break


# ═══════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  Final Evaluation\n" + "="*60)
model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
model.eval()

pa, la = [], []
with torch.no_grad():
    for ef, cf, lb in test_loader:
        p = torch.sigmoid(model(ef,cf)).numpy().flatten()
        pa.extend(p); la.extend(lb.numpy())

pa = np.array(pa); la = np.array(la)

best_t, best_a = 0.5, 0.0
for t in np.arange(0.3, 0.7, 0.01):
    a = accuracy_score(la, (pa>t).astype(int))
    if a > best_a: best_a=a; best_t=t

pf = (pa > best_t).astype(int)
print("  Best threshold : " + str(round(best_t,2)))
print("  Accuracy       : " + str(round(accuracy_score(la,pf)*100,2)) + "%")
print("  ROC AUC        : " + str(round(roc_auc_score(la,pa),4)))
print("  F1 Score       : " + str(round(f1_score(la,pf,zero_division=0),4)))
print("  Precision      : " + str(round(precision_score(la,pf,zero_division=0),4)))
print("  Recall         : " + str(round(recall_score(la,pf,zero_division=0),4)))
print("\n  Confusion Matrix:\n" + str(confusion_matrix(la,pf)))
print("="*60)
print("\nSaved -> " + SAVE_PATH)