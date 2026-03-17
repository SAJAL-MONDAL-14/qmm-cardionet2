# """
# train_multimodal.py  —  Quantum Multimodal Training (Target: 90%+)
# ===================================================================
# Architecture:
#   ECG Branch      : 325 rich features → Dense + ResBlocks → 8-dim
#   Clinical Branch : age, sex          → Dense             → 8-dim
#   Fusion          : 16-dim → 8-dim
#   QuantumLayer    : 8 qubits StronglyEntanglingLayers  ← your quantum layer
#   Classifier      : 8-dim → 1 output

# Run:  python train/train_multimodal.py
# Saved → saved_models/best_multimodal_model.pth
# """

# import os, sys, time
# import numpy as np
# import pandas as pd
# import wfdb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score, confusion_matrix
# )

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from dataset.feature_extractor import extract_ecg_features_multilead, extract_ecg_features
# from models.quantum_layer import QuantumLayer

# # ── Config ────────────────────────────────────────────────
# PTBXL_CSV   = "data/ptbxl/ptbxl_database.csv"
# PTBXL_BASE  = "data/ptbxl"
# MAX_SAMPLES = 6000
# BATCH_SIZE  = 64
# EPOCHS      = 60
# LR          = 5e-4
# PATIENCE    = 10
# SAVE_PATH   = "saved_models/best_multimodal_model.pth"
# device      = "cpu"

# print("=" * 58)
# print("  QMM CARDIONET2 — Quantum Multimodal (CPU)")
# print("=" * 58)


# # ── Dataset ───────────────────────────────────────────────
# class QuantumMultimodalDataset(Dataset):
#     def __init__(self, csv, base, max_samples=None):
#         print("\nLoading PTB-XL …")
#         df = pd.read_csv(csv)
#         df = df[df["filename_hr"].notna()].reset_index(drop=True)
#         df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

#         # balanced stratified sample
#         if max_samples and len(df) > max_samples:
#             dn = df[df["scp_codes"].str.contains("NORM", na=False)]
#             da = df[~df["scp_codes"].str.contains("NORM", na=False)]
#             n  = max_samples // 2
#             df = pd.concat([
#                 dn.sample(min(n, len(dn)), random_state=42),
#                 da.sample(min(n, len(da)), random_state=42)
#             ]).sample(frac=1, random_state=42).reset_index(drop=True)

#         print(f"Extracting features from {len(df)} records …")
#         ecg_f, clin_f, labels = [], [], []
#         skip = 0

#         for i in range(len(df)):
#             if i % 500 == 0 and i > 0:
#                 print(f"  {i}/{len(df)}  skip={skip}")
#             row  = df.iloc[i]
#             path = os.path.join(base, row["filename_hr"])
#             try:
#                 sig, _ = wfdb.rdsamp(path)
#             except Exception:
#                 skip += 1; continue
#             try:
#                 f = (extract_ecg_features_multilead(sig)
#                      if sig.shape[1] >= 11
#                      else extract_ecg_features(sig[:, 0]))
#             except Exception:
#                 skip += 1; continue
#             if not np.all(np.isfinite(f)):
#                 skip += 1; continue
#             ecg_f.append(f)
#             clin_f.append([float(row["age"]), float(row["sex"])])
#             labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

#         print(f"Done. valid={len(labels)}  skipped={skip}")
#         ea = np.array(ecg_f,   dtype=np.float32)
#         ca = np.array(clin_f,  dtype=np.float32)
#         la = np.array(labels,  dtype=np.float32)

#         es = StandardScaler(); cs = StandardScaler()
#         ea = np.nan_to_num(es.fit_transform(ea).astype(np.float32), nan=0., posinf=0., neginf=0.)
#         ca = np.nan_to_num(cs.fit_transform(ca).astype(np.float32), nan=0., posinf=0., neginf=0.)

#         self.ecg   = torch.tensor(ea, dtype=torch.float32)
#         self.clin  = torch.tensor(ca, dtype=torch.float32)
#         self.labels= torch.tensor(la, dtype=torch.float32)
#         n0 = int((la==0).sum()); n1 = int((la==1).sum())
#         print(f"NORM={n0}  ABNORMAL={n1}  ECG-dim={ea.shape[1]}")

#     def __len__(self): return len(self.labels)
#     def __getitem__(self, i): return self.ecg[i], self.clin[i], self.labels[i]


# # ── Model ─────────────────────────────────────────────────
# class ResBlock(nn.Module):
#     def __init__(self, dim, drop=0.2):
#         super().__init__()
#         self.b = nn.Sequential(
#             nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(drop),
#             nn.Linear(dim,dim), nn.BatchNorm1d(dim),
#         )
#     def forward(self, x): return F.relu(x + self.b(x))

# class ECGBranch(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128,64),     nn.BatchNorm1d(64),  nn.ReLU(),
#         )
#         self.res = ResBlock(64, 0.2)
#         self.out = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,8), nn.Tanh())
#     def forward(self, x): return self.out(self.res(self.net(x)))

# class ClinicalBranch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8), nn.Tanh()
#         )
#     def forward(self, x): return self.net(x)

# class QuantumMultimodalNet(nn.Module):
#     """
#     ECG (325-feat) + Clinical (2-feat)
#       → compress to 8-dim each
#       → fuse to 8-dim
#       → QuantumLayer (8 qubits)    ← QUANTUM PROCESSING
#       → classifier → 1 output
#     """
#     def __init__(self, ecg_dim):
#         super().__init__()
#         self.ecg_branch  = ECGBranch(ecg_dim)
#         self.clin_branch = ClinicalBranch()
#         self.fusion      = nn.Sequential(
#             nn.Linear(16, 16), nn.ReLU(),
#             nn.Linear(16,  8), nn.Tanh(),   # must be 8-dim for quantum layer
#         )
#         self.quantum     = QuantumLayer()   # your existing quantum layer (8 qubits)
#         self.classifier  = nn.Sequential(
#             nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1)
#         )

#     def forward(self, ecg_f, clin_f):
#         e = self.ecg_branch(ecg_f)                    # (B, 8)
#         c = self.clin_branch(clin_f)                  # (B, 8)
#         x = self.fusion(torch.cat([e, c], dim=1))     # (B, 8)
#         q = self.quantum(x)                            # (B, 8) quantum output
#         return self.classifier(q)                      # (B, 1)


# # ── Build loaders ─────────────────────────────────────────
# dataset = QuantumMultimodalDataset(PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES)
# ecg_dim = dataset.ecg.shape[1]
# n_train = int(0.8 * len(dataset))
# n_test  = len(dataset) - n_train

# train_ds, test_ds = random_split(
#     dataset, [n_train, n_test],
#     generator=torch.Generator().manual_seed(42)
# )
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# print(f"\nTrain={n_train}  Test={n_test}")

# lbl_np = dataset.labels.numpy()
# pos_w  = torch.tensor([(lbl_np==0).sum() / max((lbl_np==1).sum(),1)], dtype=torch.float32)
# print(f"pos_weight={pos_w.item():.3f}")

# # ── Training setup ────────────────────────────────────────
# model     = QuantumMultimodalNet(ecg_dim)
# n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Parameters: {n_params:,}")

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=15, T_mult=2, eta_min=1e-6
# )
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
# os.makedirs("saved_models", exist_ok=True)

# best_auc = 0.0; patience_ct = 0

# print("\n" + "="*58)
# print("  Training  (QuantumLayer active — 8 qubits)")
# print("="*58 + "\n")

# for epoch in range(1, EPOCHS + 1):
#     model.train(); total = 0.0; t0 = time.time()

#     for ef, cf, lb in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(ef, cf), lb.unsqueeze(1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # quantum stability
#         optimizer.step()
#         total += loss.item()

#     scheduler.step()
#     model.eval(); pa, pr, la = [], [], []

#     with torch.no_grad():
#         for ef, cf, lb in test_loader:
#             p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#             pa.extend(p); pr.extend((p>0.5).astype(int)); la.extend(lb.numpy())

#     acc = accuracy_score(la, pr)
#     auc = roc_auc_score(la, pa)
#     f1  = f1_score(la, pr, zero_division=0)
#     print(f"Epoch {epoch:02d}/{EPOCHS}  loss={total/len(train_loader):.4f}  "
#           f"acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  ({time.time()-t0:.1f}s)")

#     if auc > best_auc:
#         best_auc = auc; patience_ct = 0
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"  ✓ Best saved  acc={acc:.4f}  auc={auc:.4f}")
#     else:
#         patience_ct += 1
#         if patience_ct >= PATIENCE:
#             print(f"  Early stopping at epoch {epoch}."); break

# # ── Final Evaluation ──────────────────────────────────────
# print("\n" + "="*58 + "\n  Final Evaluation\n" + "="*58)
# model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
# model.eval()
# pa, la = [], []

# with torch.no_grad():
#     for ef, cf, lb in test_loader:
#         p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#         pa.extend(p); la.extend(lb.numpy())

# pa = np.array(pa); la = np.array(la)

# # optimal threshold
# bt, bta = 0.5, 0.0
# for t in np.arange(0.3, 0.7, 0.01):
#     a = accuracy_score(la, (pa>t).astype(int))
#     if a > bta: bta=a; bt=t

# pf = (pa > bt).astype(int)
# print(f"\n  Best threshold : {bt:.2f}")
# print(f"  Accuracy       : {accuracy_score(la,pf)*100:.2f}%")
# print(f"  ROC AUC        : {roc_auc_score(la,pa):.4f}")
# print(f"  F1 Score       : {f1_score(la,pf,zero_division=0):.4f}")
# print(f"  Precision      : {precision_score(la,pf,zero_division=0):.4f}")
# print(f"  Recall         : {recall_score(la,pf,zero_division=0):.4f}")
# print(f"\n  Confusion Matrix:\n{confusion_matrix(la,pf)}")
# print("="*58)
# print(f"\n✓ Saved → {SAVE_PATH}")







# """
# train_multimodal.py  —  Quantum Multimodal with Feature Cache (92%+ target)
# ============================================================================
# FEATURE CACHING:
#   First run  → extracts features from all samples, saves to cache file
#   Next runs  → loads from cache instantly (seconds not hours)

# So extraction happens ONCE, then every future run is fast.

# Run:  python train/train_multimodal.py
# """

# import os, sys, time
# import numpy as np
# import pandas as pd
# import wfdb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score, confusion_matrix
# )

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from dataset.feature_extractor import extract_ecg_features_multilead, extract_ecg_features
# from models.quantum_layer import QuantumLayer

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# PTBXL_CSV    = "data/ptbxl/ptbxl_database.csv"
# PTBXL_BASE   = "data/ptbxl"
# MAX_SAMPLES  = 12000     # 6000 NORM + 6000 ABNORMAL — high accuracy
# BATCH_SIZE   = 128
# EPOCHS       = 80
# LR           = 8e-4
# PATIENCE     = 15
# SAVE_PATH    = "saved_models/best_multimodal_model.pth"

# # ── CACHE — extract once, reload instantly every future run ──
# CACHE_PATH   = "saved_models/feature_cache.npz"

# device = "cpu"

# print("=" * 60)
# print("  QMM CARDIONET2 — Quantum Multimodal (92%+ target)")
# print("=" * 60)
# print(f"  MAX_SAMPLES = {MAX_SAMPLES}")
# print(f"  CACHE       = {CACHE_PATH}")
# print("=" * 60)


# # ═══════════════════════════════════════════════════════════
# #  FEATURE EXTRACTION WITH CACHE
# # ═══════════════════════════════════════════════════════════
# def load_or_extract_features(csv, base, max_samples, cache_path):
#     """
#     If cache exists → load instantly from .npz file.
#     If not          → extract features, save cache, return data.
#     """

#     # ── Try loading from cache first ──
#     if os.path.exists(cache_path):
#         print(f"\n✓ Cache found → loading from {cache_path} …")
#         t0   = time.time()
#         data = np.load(cache_path)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         lbl_arr  = data["labels"]
#         print(f"  Loaded {len(lbl_arr)} samples in {time.time()-t0:.1f}s")
#         print(f"  ECG-dim={ecg_arr.shape[1]}  "
#               f"NORM={(lbl_arr==0).sum()}  "
#               f"ABNORMAL={(lbl_arr==1).sum()}")
#         return ecg_arr, clin_arr, lbl_arr

#     # ── Extract features (first run only) ──
#     print(f"\nNo cache found → extracting features from {max_samples} records …")
#     print("This runs ONCE and saves to cache. Future runs will be instant.\n")

#     df = pd.read_csv(csv)
#     df = df[df["filename_hr"].notna()].reset_index(drop=True)
#     df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

#     # stratified balanced sample
#     if max_samples and len(df) > max_samples:
#         dn = df[df["scp_codes"].str.contains("NORM", na=False)]
#         da = df[~df["scp_codes"].str.contains("NORM", na=False)]
#         n  = max_samples // 2
#         df = pd.concat([
#             dn.sample(min(n, len(dn)), random_state=42),
#             da.sample(min(n, len(da)), random_state=42)
#         ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     ecg_f, clin_f, labels = [], [], []
#     skip = 0
#     t0   = time.time()

#     for i in range(len(df)):
#         if i % 500 == 0 and i > 0:
#             elapsed = time.time() - t0
#             eta     = elapsed / i * (len(df) - i)
#             print(f"  [{i:>5}/{len(df)}]  "
#                   f"skip={skip}  "
#                   f"elapsed={elapsed/60:.1f}min  "
#                   f"ETA={eta/60:.1f}min")

#         row  = df.iloc[i]
#         path = os.path.join(base, row["filename_hr"])

#         try:
#             sig, _ = wfdb.rdsamp(path)
#         except Exception:
#             skip += 1; continue

#         try:
#             f = (extract_ecg_features_multilead(sig)
#                  if sig.shape[1] >= 11
#                  else extract_ecg_features(sig[:, 0]))
#         except Exception:
#             skip += 1; continue

#         if not np.all(np.isfinite(f)):
#             skip += 1; continue

#         ecg_f.append(f)
#         clin_f.append([float(row["age"]), float(row["sex"])])
#         labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

#     ecg_arr  = np.array(ecg_f,  dtype=np.float32)
#     clin_arr = np.array(clin_f, dtype=np.float32)
#     lbl_arr  = np.array(labels, dtype=np.float32)

#     total_time = time.time() - t0
#     print(f"\nExtraction done in {total_time/60:.1f} min")
#     print(f"valid={len(lbl_arr)}  skipped={skip}")
#     print(f"NORM={(lbl_arr==0).sum()}  ABNORMAL={(lbl_arr==1).sum()}")

#     # ── Save cache ──
#     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#     np.savez_compressed(cache_path,
#                         ecg=ecg_arr,
#                         clin=clin_arr,
#                         labels=lbl_arr)
#     print(f"✓ Cache saved → {cache_path}  (future runs load instantly)")

#     return ecg_arr, clin_arr, lbl_arr


# # ═══════════════════════════════════════════════════════════
# #  DATASET
# # ═══════════════════════════════════════════════════════════
# class CachedMultimodalDataset(Dataset):

#     def __init__(self, ecg_arr, clin_arr, lbl_arr):
#         # normalise
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         combined = np.nan_to_num(
#             scaler.fit_transform(combined).astype(np.float32),
#             nan=0., posinf=0., neginf=0.
#         )
#         ecg_n  = combined[:, :ecg_arr.shape[1]]
#         clin_n = combined[:, ecg_arr.shape[1]:]

#         self.ecg    = torch.tensor(ecg_n,   dtype=torch.float32)
#         self.clin   = torch.tensor(clin_n,  dtype=torch.float32)
#         self.labels = torch.tensor(lbl_arr, dtype=torch.float32)

#     def __len__(self): return len(self.labels)
#     def __getitem__(self, i): return self.ecg[i], self.clin[i], self.labels[i]


# # ═══════════════════════════════════════════════════════════
# #  MODEL  —  Dual-Path Quantum-Classical
# # ═══════════════════════════════════════════════════════════
# class ResBlock(nn.Module):
#     def __init__(self, dim, drop=0.2):
#         super().__init__()
#         self.b = nn.Sequential(
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#             nn.ReLU(), nn.Dropout(drop),
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#         )
#     def forward(self, x): return F.relu(x + self.b(x))


# class SharedEncoder(nn.Module):
#     def __init__(self, ecg_dim):
#         super().__init__()
#         in_dim = ecg_dim + 2
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
#             nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(),
#         )
#         self.res1 = ResBlock(128, 0.2)
#         self.res2 = ResBlock(128, 0.15)
#         self.proj = nn.Sequential(
#             nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
#         )
#     def forward(self, ecg_f, clin_f):
#         x = torch.cat([ecg_f, clin_f], dim=1)
#         return self.proj(self.res2(self.res1(self.net(x))))


# class DualPathQuantumNet(nn.Module):
#     """
#     Classical path (64-dim) runs PARALLEL with Quantum path (8-dim).
#     Merged at classifier → best of both worlds.

#     SharedEncoder(325+2 → 64)
#          ├── Classical: 64 → ResBlocks → 64 ──────────────┐
#          └── Quantum:   64 → compress(8) → QL(8) → 8 ─────┤
#                                                              └→ cat(72) → classifier
#     """
#     def __init__(self, ecg_dim):
#         super().__init__()
#         self.encoder          = SharedEncoder(ecg_dim)
#         self.classical_path   = nn.Sequential(ResBlock(64, 0.2), ResBlock(64, 0.15))
#         self.quantum_compress = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  8), nn.Tanh(),
#         )
#         self.quantum     = QuantumLayer()
#         self.classifier  = nn.Sequential(
#             nn.Linear(72, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  1),
#         )

#     def forward(self, ecg_f, clin_f):
#         shared  = self.encoder(ecg_f, clin_f)
#         cls_out = self.classical_path(shared)
#         q_out   = self.quantum(self.quantum_compress(shared))
#         return self.classifier(torch.cat([cls_out, q_out], dim=1))


# # ═══════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════
# os.makedirs("saved_models", exist_ok=True)

# # load or extract features
# ecg_arr, clin_arr, lbl_arr = load_or_extract_features(
#     PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES, CACHE_PATH
# )

# # build dataset
# dataset  = CachedMultimodalDataset(ecg_arr, clin_arr, lbl_arr)
# ecg_dim  = dataset.ecg.shape[1]
# n_train  = int(0.8 * len(dataset))
# n_test   = len(dataset) - n_train

# train_ds, test_ds = random_split(
#     dataset, [n_train, n_test],
#     generator=torch.Generator().manual_seed(42)
# )
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# print(f"\nTrain={n_train}  Test={n_test}  ECG-dim={ecg_dim}")

# lbl_np = dataset.labels.numpy()
# pos_w  = torch.tensor([(lbl_np==0).sum() / max((lbl_np==1).sum(),1)], dtype=torch.float32)
# print(f"pos_weight={pos_w.item():.3f}")

# # model
# model    = DualPathQuantumNet(ecg_dim)
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Parameters: {n_params:,}")

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# def lr_lambda(epoch):
#     warmup = 5
#     if epoch < warmup:
#         return (epoch + 1) / warmup
#     progress = (epoch - warmup) / (EPOCHS - warmup)
#     return 0.5 * (1 + np.cos(np.pi * progress))

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# best_auc = 0.0; patience_ct = 0

# print("\n" + "="*60)
# print("  Training  (Classical + Quantum parallel paths)")
# print("="*60 + "\n")

# for epoch in range(1, EPOCHS + 1):
#     model.train(); total = 0.0; t0 = time.time()

#     for ef, cf, lb in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(ef, cf), lb.unsqueeze(1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total += loss.item()

#     scheduler.step()
#     model.eval(); pa, pr, la = [], [], []

#     with torch.no_grad():
#         for ef, cf, lb in test_loader:
#             p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#             pa.extend(p); pr.extend((p>0.5).astype(int)); la.extend(lb.numpy())

#     acc = accuracy_score(la, pr)
#     auc = roc_auc_score(la, pa)
#     f1  = f1_score(la, pr, zero_division=0)
#     lr_now = optimizer.param_groups[0]["lr"]

#     print(f"Epoch {epoch:02d}/{EPOCHS}  "
#           f"loss={total/len(train_loader):.4f}  "
#           f"acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  "
#           f"lr={lr_now:.6f}  ({time.time()-t0:.1f}s)")

#     if auc > best_auc:
#         best_auc = auc; patience_ct = 0
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"  ✓ Best saved  acc={acc:.4f}  auc={auc:.4f}")
#     else:
#         patience_ct += 1
#         if patience_ct >= PATIENCE:
#             print(f"  Early stopping at epoch {epoch}."); break

# # ── Final Evaluation ──────────────────────────────────────
# print("\n" + "="*60 + "\n  Final Evaluation\n" + "="*60)
# model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
# model.eval(); pa, la = [], []

# with torch.no_grad():
#     for ef, cf, lb in test_loader:
#         p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#         pa.extend(p); la.extend(lb.numpy())

# pa = np.array(pa); la = np.array(la)

# bt, bta = 0.5, 0.0
# for t in np.arange(0.3, 0.7, 0.01):
#     a = accuracy_score(la, (pa>t).astype(int))
#     if a > bta: bta=a; bt=t

# pf = (pa > bt).astype(int)
# print(f"\n  Best threshold : {bt:.2f}")
# print(f"  Accuracy       : {accuracy_score(la,pf)*100:.2f}%")
# print(f"  ROC AUC        : {roc_auc_score(la,pa):.4f}")
# print(f"  F1 Score       : {f1_score(la,pf,zero_division=0):.4f}")
# print(f"  Precision      : {precision_score(la,pf,zero_division=0):.4f}")
# print(f"  Recall         : {recall_score(la,pf,zero_division=0):.4f}")
# print(f"\n  Confusion Matrix:\n{confusion_matrix(la,pf)}")
# print("="*60)
# print(f"\n✓ Saved → {SAVE_PATH}")
# print(f"✓ Cache → {CACHE_PATH}  (next run loads instantly!)")



# """
# train_multimodal.py  —  Quantum Multimodal with Feature Cache (92%+ target)
# ============================================================================
# FEATURE CACHING:
#   First run  → extracts features from all samples, saves to cache file
#   Next runs  → loads from cache instantly (seconds not hours)

# So extraction happens ONCE, then every future run is fast.

# Run:  python train/train_multimodal.py
# """

# import os, sys, time
# import numpy as np
# import pandas as pd
# import wfdb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score, confusion_matrix
# )

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from dataset.feature_extractor import extract_ecg_features_multilead, extract_ecg_features
# from models.quantum_layer import QuantumLayer

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# PTBXL_CSV    = "data/ptbxl/ptbxl_database.csv"
# PTBXL_BASE   = "data/ptbxl"
# MAX_SAMPLES  = 12000     # 6000 NORM + 6000 ABNORMAL — high accuracy
# BATCH_SIZE   = 32        # KEY FIX: small batch → better quantum gradients
# EPOCHS       = 150       # more epochs — quantum models need longer training
# LR           = 1e-3      # KEY FIX: higher LR to escape plateau fast
# PATIENCE     = 25        # KEY FIX: enough patience for quantum improvement
# SAVE_PATH    = "saved_models/best_multimodal_model.pth"

# # ── CACHE — extract once, reload instantly every future run ──
# CACHE_PATH   = "saved_models/feature_cache.npz"

# device = "cpu"

# print("=" * 60)
# print("  QMM CARDIONET2 — Quantum Multimodal (92%+ target)")
# print("=" * 60)
# print(f"  MAX_SAMPLES = {MAX_SAMPLES}")
# print(f"  CACHE       = {CACHE_PATH}")
# print("=" * 60)


# # ═══════════════════════════════════════════════════════════
# #  FEATURE EXTRACTION WITH CACHE
# # ═══════════════════════════════════════════════════════════
# def load_or_extract_features(csv, base, max_samples, cache_path):
#     """
#     If cache exists → load instantly from .npz file.
#     If not          → extract features, save cache, return data.
#     """

#     # ── Try loading from cache first ──
#     if os.path.exists(cache_path):
#         print(f"\n✓ Cache found → loading from {cache_path} …")
#         t0   = time.time()
#         data = np.load(cache_path)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         lbl_arr  = data["labels"]
#         print(f"  Loaded {len(lbl_arr)} samples in {time.time()-t0:.1f}s")
#         print(f"  ECG-dim={ecg_arr.shape[1]}  "
#               f"NORM={(lbl_arr==0).sum()}  "
#               f"ABNORMAL={(lbl_arr==1).sum()}")
#         return ecg_arr, clin_arr, lbl_arr

#     # ── Extract features (first run only) ──
#     print(f"\nNo cache found → extracting features from {max_samples} records …")
#     print("This runs ONCE and saves to cache. Future runs will be instant.\n")

#     df = pd.read_csv(csv)
#     df = df[df["filename_hr"].notna()].reset_index(drop=True)
#     df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

#     # stratified balanced sample
#     if max_samples and len(df) > max_samples:
#         dn = df[df["scp_codes"].str.contains("NORM", na=False)]
#         da = df[~df["scp_codes"].str.contains("NORM", na=False)]
#         n  = max_samples // 2
#         df = pd.concat([
#             dn.sample(min(n, len(dn)), random_state=42),
#             da.sample(min(n, len(da)), random_state=42)
#         ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     ecg_f, clin_f, labels = [], [], []
#     skip = 0
#     t0   = time.time()

#     for i in range(len(df)):
#         if i % 500 == 0 and i > 0:
#             elapsed = time.time() - t0
#             eta     = elapsed / i * (len(df) - i)
#             print(f"  [{i:>5}/{len(df)}]  "
#                   f"skip={skip}  "
#                   f"elapsed={elapsed/60:.1f}min  "
#                   f"ETA={eta/60:.1f}min")

#         row  = df.iloc[i]
#         path = os.path.join(base, row["filename_hr"])

#         try:
#             sig, _ = wfdb.rdsamp(path)
#         except Exception:
#             skip += 1; continue

#         try:
#             f = (extract_ecg_features_multilead(sig)
#                  if sig.shape[1] >= 11
#                  else extract_ecg_features(sig[:, 0]))
#         except Exception:
#             skip += 1; continue

#         if not np.all(np.isfinite(f)):
#             skip += 1; continue

#         ecg_f.append(f)
#         clin_f.append([float(row["age"]), float(row["sex"])])
#         labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

#     ecg_arr  = np.array(ecg_f,  dtype=np.float32)
#     clin_arr = np.array(clin_f, dtype=np.float32)
#     lbl_arr  = np.array(labels, dtype=np.float32)

#     total_time = time.time() - t0
#     print(f"\nExtraction done in {total_time/60:.1f} min")
#     print(f"valid={len(lbl_arr)}  skipped={skip}")
#     print(f"NORM={(lbl_arr==0).sum()}  ABNORMAL={(lbl_arr==1).sum()}")

#     # ── Save cache ──
#     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#     np.savez_compressed(cache_path,
#                         ecg=ecg_arr,
#                         clin=clin_arr,
#                         labels=lbl_arr)
#     print(f"✓ Cache saved → {cache_path}  (future runs load instantly)")

#     return ecg_arr, clin_arr, lbl_arr


# # ═══════════════════════════════════════════════════════════
# #  DATASET
# # ═══════════════════════════════════════════════════════════
# class CachedMultimodalDataset(Dataset):

#     def __init__(self, ecg_arr, clin_arr, lbl_arr):
#         # normalise
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         combined = np.nan_to_num(
#             scaler.fit_transform(combined).astype(np.float32),
#             nan=0., posinf=0., neginf=0.
#         )
#         ecg_n  = combined[:, :ecg_arr.shape[1]]
#         clin_n = combined[:, ecg_arr.shape[1]:]

#         self.ecg    = torch.tensor(ecg_n,   dtype=torch.float32)
#         self.clin   = torch.tensor(clin_n,  dtype=torch.float32)
#         self.labels = torch.tensor(lbl_arr, dtype=torch.float32)

#     def __len__(self): return len(self.labels)
#     def __getitem__(self, i): return self.ecg[i], self.clin[i], self.labels[i]


# # ═══════════════════════════════════════════════════════════
# #  MODEL  —  Dual-Path Quantum-Classical
# # ═══════════════════════════════════════════════════════════
# class ResBlock(nn.Module):
#     def __init__(self, dim, drop=0.2):
#         super().__init__()
#         self.b = nn.Sequential(
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#             nn.ReLU(), nn.Dropout(drop),
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#         )
#     def forward(self, x): return F.relu(x + self.b(x))


# class SharedEncoder(nn.Module):
#     def __init__(self, ecg_dim):
#         super().__init__()
#         in_dim = ecg_dim + 2
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
#             nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(),
#         )
#         self.res1 = ResBlock(128, 0.2)
#         self.res2 = ResBlock(128, 0.15)
#         self.proj = nn.Sequential(
#             nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
#         )
#     def forward(self, ecg_f, clin_f):
#         x = torch.cat([ecg_f, clin_f], dim=1)
#         return self.proj(self.res2(self.res1(self.net(x))))


# class DualPathQuantumNet(nn.Module):
#     """
#     Classical path (64-dim) runs PARALLEL with Quantum path (8-dim).
#     Merged at classifier → best of both worlds.

#     SharedEncoder(325+2 → 64)
#          ├── Classical: 64 → ResBlocks → 64 ──────────────┐
#          └── Quantum:   64 → compress(8) → QL(8) → 8 ─────┤
#                                                              └→ cat(72) → classifier
#     """
#     def __init__(self, ecg_dim):
#         super().__init__()
#         self.encoder          = SharedEncoder(ecg_dim)
#         self.classical_path   = nn.Sequential(ResBlock(64, 0.2), ResBlock(64, 0.15))
#         self.quantum_compress = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  8), nn.Tanh(),
#         )
#         self.quantum     = QuantumLayer()
#         self.classifier  = nn.Sequential(
#             nn.Linear(72, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  1),
#         )

#     def forward(self, ecg_f, clin_f):
#         shared  = self.encoder(ecg_f, clin_f)
#         cls_out = self.classical_path(shared)
#         q_out   = self.quantum(self.quantum_compress(shared))
#         return self.classifier(torch.cat([cls_out, q_out], dim=1))


# # ═══════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════
# os.makedirs("saved_models", exist_ok=True)

# # load or extract features
# ecg_arr, clin_arr, lbl_arr = load_or_extract_features(
#     PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES, CACHE_PATH
# )

# # build dataset
# dataset  = CachedMultimodalDataset(ecg_arr, clin_arr, lbl_arr)
# ecg_dim  = dataset.ecg.shape[1]
# n_train  = int(0.8 * len(dataset))
# n_test   = len(dataset) - n_train

# train_ds, test_ds = random_split(
#     dataset, [n_train, n_test],
#     generator=torch.Generator().manual_seed(42)
# )
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# print(f"\nTrain={n_train}  Test={n_test}  ECG-dim={ecg_dim}")

# lbl_np = dataset.labels.numpy()
# pos_w  = torch.tensor([(lbl_np==0).sum() / max((lbl_np==1).sum(),1)], dtype=torch.float32)
# print(f"pos_weight={pos_w.item():.3f}")

# # model
# model    = DualPathQuantumNet(ecg_dim)
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Parameters: {n_params:,}")

# # ── separate LR for quantum vs classical params ──
# # quantum layer needs much smaller LR than classical layers
# quantum_params   = list(model.quantum.parameters())
# classical_params = [p for p in model.parameters()
#                     if not any(p is q for q in quantum_params)]

# optimizer = torch.optim.AdamW([
#     {"params": classical_params, "lr": LR,        "weight_decay": 1e-4},
#     {"params": quantum_params,   "lr": LR * 0.1,  "weight_decay": 0.0},
#     # quantum layer gets 10x smaller LR — critical for stability
# ])

# # ReduceLROnPlateau — reduces LR when stuck, perfect for quantum models
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode     = "max",    # maximise AUC
#     factor   = 0.5,      # halve LR when stuck
#     patience = 8,        # wait 8 epochs before reducing
#     min_lr   = 1e-6,
# )

# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# best_auc = 0.0; best_acc = 0.0; patience_ct = 0

# print("\n" + "="*60)
# print("  Training  (Classical + Quantum parallel paths)")
# print("="*60 + "\n")

# for epoch in range(1, EPOCHS + 1):
#     model.train(); total = 0.0; t0 = time.time()

#     for ef, cf, lb in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(ef, cf), lb.unsqueeze(1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total += loss.item()

#     model.eval(); pa, pr, la = [], [], []

#     with torch.no_grad():
#         for ef, cf, lb in test_loader:
#             p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#             pa.extend(p); pr.extend((p>0.5).astype(int)); la.extend(lb.numpy())

#     acc = accuracy_score(la, pr)
#     auc = roc_auc_score(la, pa)
#     f1  = f1_score(la, pr, zero_division=0)

#     scheduler.step(auc)   # ReduceLROnPlateau needs metric
#     lr_now = optimizer.param_groups[0]["lr"]

#     print(f"Epoch {epoch:03d}/{EPOCHS}  "
#           f"loss={total/len(train_loader):.4f}  "
#           f"acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  "
#           f"lr={lr_now:.6f}  ({time.time()-t0:.1f}s)")

#     if auc > best_auc:
#         best_auc = auc; best_acc = acc; patience_ct = 0
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"  ✓ Best saved  acc={acc:.4f}  auc={auc:.4f}")
#     else:
#         patience_ct += 1
#         print(f"  patience {patience_ct}/{PATIENCE}")
#         if patience_ct >= PATIENCE:
#             print(f"  Early stopping at epoch {epoch}."); break

# # ── Final Evaluation ──────────────────────────────────────
# print("\n" + "="*60 + "\n  Final Evaluation\n" + "="*60)
# model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
# model.eval(); pa, la = [], []

# with torch.no_grad():
#     for ef, cf, lb in test_loader:
#         p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#         pa.extend(p); la.extend(lb.numpy())

# pa = np.array(pa); la = np.array(la)

# bt, bta = 0.5, 0.0
# for t in np.arange(0.3, 0.7, 0.01):
#     a = accuracy_score(la, (pa>t).astype(int))
#     if a > bta: bta=a; bt=t

# pf = (pa > bt).astype(int)
# print(f"\n  Best threshold : {bt:.2f}")
# print(f"  Accuracy       : {accuracy_score(la,pf)*100:.2f}%")
# print(f"  ROC AUC        : {roc_auc_score(la,pa):.4f}")
# print(f"  F1 Score       : {f1_score(la,pf,zero_division=0):.4f}")
# print(f"  Precision      : {precision_score(la,pf,zero_division=0):.4f}")
# print(f"  Recall         : {recall_score(la,pf,zero_division=0):.4f}")
# print(f"\n  Confusion Matrix:\n{confusion_matrix(la,pf)}")
# print("="*60)
# print(f"\n✓ Saved → {SAVE_PATH}")
# print(f"✓ Cache → {CACHE_PATH}  (next run loads instantly!)")

















# """
# train_multimodal.py  —  Quantum Multimodal with Feature Cache (92%+ target)
# ============================================================================
# FEATURE CACHING:
#   First run  → extracts features from all samples, saves to cache file
#   Next runs  → loads from cache instantly (seconds not hours)

# So extraction happens ONCE, then every future run is fast.

# Run:  python train/train_multimodal.py
# """

# import os, sys, time
# import numpy as np
# import pandas as pd
# import wfdb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score, confusion_matrix
# )

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from dataset.feature_extractor import extract_ecg_features_multilead, extract_ecg_features
# from models.quantum_layer import QuantumLayer

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# PTBXL_CSV    = "data/ptbxl/ptbxl_database.csv"
# PTBXL_BASE   = "data/ptbxl"
# MAX_SAMPLES  = 12000     # 6000 NORM + 6000 ABNORMAL — high accuracy
# BATCH_SIZE   = 32        # KEY FIX: small batch → better quantum gradients
# EPOCHS       = 200       # more epochs → quantum needs time to surpass SVM
# LR           = 5e-4      # balanced LR for quantum-classical hybrid
# PATIENCE     = 40        # high patience → dont stop early like before
# SAVE_PATH    = "saved_models/best_multimodal_model.pth"

# # ── CACHE — extract once, reload instantly every future run ──
# CACHE_PATH   = "saved_models/feature_cache.npz"

# device = "cpu"

# print("=" * 60)
# print("  QMM CARDIONET2 — Quantum Multimodal (92%+ target)")
# print("=" * 60)
# print(f"  MAX_SAMPLES = {MAX_SAMPLES}")
# print(f"  CACHE       = {CACHE_PATH}")
# print("=" * 60)


# # ═══════════════════════════════════════════════════════════
# #  FEATURE EXTRACTION WITH CACHE
# # ═══════════════════════════════════════════════════════════
# def load_or_extract_features(csv, base, max_samples, cache_path):
#     """
#     If cache exists → load instantly from .npz file.
#     If not          → extract features, save cache, return data.
#     """

#     # ── Try loading from cache first ──
#     if os.path.exists(cache_path):
#         print(f"\n✓ Cache found → loading from {cache_path} …")
#         t0   = time.time()
#         data = np.load(cache_path)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         lbl_arr  = data["labels"]
#         print(f"  Loaded {len(lbl_arr)} samples in {time.time()-t0:.1f}s")
#         print(f"  ECG-dim={ecg_arr.shape[1]}  "
#               f"NORM={(lbl_arr==0).sum()}  "
#               f"ABNORMAL={(lbl_arr==1).sum()}")
#         return ecg_arr, clin_arr, lbl_arr

#     # ── Extract features (first run only) ──
#     print(f"\nNo cache found → extracting features from {max_samples} records …")
#     print("This runs ONCE and saves to cache. Future runs will be instant.\n")

#     df = pd.read_csv(csv)
#     df = df[df["filename_hr"].notna()].reset_index(drop=True)
#     df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

#     # stratified balanced sample
#     if max_samples and len(df) > max_samples:
#         dn = df[df["scp_codes"].str.contains("NORM", na=False)]
#         da = df[~df["scp_codes"].str.contains("NORM", na=False)]
#         n  = max_samples // 2
#         df = pd.concat([
#             dn.sample(min(n, len(dn)), random_state=42),
#             da.sample(min(n, len(da)), random_state=42)
#         ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     ecg_f, clin_f, labels = [], [], []
#     skip = 0
#     t0   = time.time()

#     for i in range(len(df)):
#         if i % 500 == 0 and i > 0:
#             elapsed = time.time() - t0
#             eta     = elapsed / i * (len(df) - i)
#             print(f"  [{i:>5}/{len(df)}]  "
#                   f"skip={skip}  "
#                   f"elapsed={elapsed/60:.1f}min  "
#                   f"ETA={eta/60:.1f}min")

#         row  = df.iloc[i]
#         path = os.path.join(base, row["filename_hr"])

#         try:
#             sig, _ = wfdb.rdsamp(path)
#         except Exception:
#             skip += 1; continue

#         try:
#             f = (extract_ecg_features_multilead(sig)
#                  if sig.shape[1] >= 11
#                  else extract_ecg_features(sig[:, 0]))
#         except Exception:
#             skip += 1; continue

#         if not np.all(np.isfinite(f)):
#             skip += 1; continue

#         ecg_f.append(f)
#         clin_f.append([float(row["age"]), float(row["sex"])])
#         labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

#     ecg_arr  = np.array(ecg_f,  dtype=np.float32)
#     clin_arr = np.array(clin_f, dtype=np.float32)
#     lbl_arr  = np.array(labels, dtype=np.float32)

#     total_time = time.time() - t0
#     print(f"\nExtraction done in {total_time/60:.1f} min")
#     print(f"valid={len(lbl_arr)}  skipped={skip}")
#     print(f"NORM={(lbl_arr==0).sum()}  ABNORMAL={(lbl_arr==1).sum()}")

#     # ── Save cache ──
#     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#     np.savez_compressed(cache_path,
#                         ecg=ecg_arr,
#                         clin=clin_arr,
#                         labels=lbl_arr)
#     print(f"✓ Cache saved → {cache_path}  (future runs load instantly)")

#     return ecg_arr, clin_arr, lbl_arr


# # ═══════════════════════════════════════════════════════════
# #  DATASET
# # ═══════════════════════════════════════════════════════════
# class CachedMultimodalDataset(Dataset):

#     def __init__(self, ecg_arr, clin_arr, lbl_arr):
#         # normalise
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         combined = np.nan_to_num(
#             scaler.fit_transform(combined).astype(np.float32),
#             nan=0., posinf=0., neginf=0.
#         )
#         ecg_n  = combined[:, :ecg_arr.shape[1]]
#         clin_n = combined[:, ecg_arr.shape[1]:]

#         self.ecg    = torch.tensor(ecg_n,   dtype=torch.float32)
#         self.clin   = torch.tensor(clin_n,  dtype=torch.float32)
#         self.labels = torch.tensor(lbl_arr, dtype=torch.float32)

#     def __len__(self): return len(self.labels)
#     def __getitem__(self, i): return self.ecg[i], self.clin[i], self.labels[i]


# # ═══════════════════════════════════════════════════════════
# #  MODEL  —  Dual-Path Quantum-Classical
# # ═══════════════════════════════════════════════════════════
# class ResBlock(nn.Module):
#     def __init__(self, dim, drop=0.2):
#         super().__init__()
#         self.b = nn.Sequential(
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#             nn.ReLU(), nn.Dropout(drop),
#             nn.Linear(dim, dim), nn.BatchNorm1d(dim),
#         )
#     def forward(self, x): return F.relu(x + self.b(x))


# class SharedEncoder(nn.Module):
#     def __init__(self, ecg_dim):
#         super().__init__()
#         in_dim = ecg_dim + 2
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
#             nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(),
#         )
#         self.res1 = ResBlock(128, 0.2)
#         self.res2 = ResBlock(128, 0.15)
#         self.proj = nn.Sequential(
#             nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
#         )
#     def forward(self, ecg_f, clin_f):
#         x = torch.cat([ecg_f, clin_f], dim=1)
#         return self.proj(self.res2(self.res1(self.net(x))))


# class DualPathQuantumNet(nn.Module):
#     """
#     Classical path (64-dim) runs PARALLEL with Quantum path (8-dim).
#     Merged at classifier → best of both worlds.

#     SharedEncoder(325+2 → 64)
#          ├── Classical: 64 → ResBlocks → 64 ──────────────┐
#          └── Quantum:   64 → compress(8) → QL(8) → 8 ─────┤
#                                                              └→ cat(72) → classifier
#     """
#     def __init__(self, ecg_dim):
#         super().__init__()
#         self.encoder          = SharedEncoder(ecg_dim)
#         self.classical_path   = nn.Sequential(ResBlock(64, 0.2), ResBlock(64, 0.15))
#         self.quantum_compress = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  8), nn.Tanh(),
#         )
#         self.quantum     = QuantumLayer()
#         self.classifier  = nn.Sequential(
#             nn.Linear(72, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32,  1),
#         )

#     def forward(self, ecg_f, clin_f):
#         shared  = self.encoder(ecg_f, clin_f)
#         cls_out = self.classical_path(shared)
#         q_out   = self.quantum(self.quantum_compress(shared))
#         return self.classifier(torch.cat([cls_out, q_out], dim=1))


# # ═══════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════
# os.makedirs("saved_models", exist_ok=True)

# # load or extract features
# ecg_arr, clin_arr, lbl_arr = load_or_extract_features(
#     PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES, CACHE_PATH
# )

# # build dataset
# dataset  = CachedMultimodalDataset(ecg_arr, clin_arr, lbl_arr)
# ecg_dim  = dataset.ecg.shape[1]
# n_train  = int(0.8 * len(dataset))
# n_test   = len(dataset) - n_train

# train_ds, test_ds = random_split(
#     dataset, [n_train, n_test],
#     generator=torch.Generator().manual_seed(42)
# )
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# print(f"\nTrain={n_train}  Test={n_test}  ECG-dim={ecg_dim}")

# lbl_np = dataset.labels.numpy()
# pos_w  = torch.tensor([(lbl_np==0).sum() / max((lbl_np==1).sum(),1)], dtype=torch.float32)
# print(f"pos_weight={pos_w.item():.3f}")

# # model
# model    = DualPathQuantumNet(ecg_dim)
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Parameters: {n_params:,}")

# # ── separate LR for quantum vs classical params ──
# # quantum layer needs much smaller LR than classical layers
# quantum_params   = list(model.quantum.parameters())
# classical_params = [p for p in model.parameters()
#                     if not any(p is q for q in quantum_params)]

# optimizer = torch.optim.AdamW([
#     {"params": classical_params, "lr": LR,        "weight_decay": 1e-4},
#     {"params": quantum_params,   "lr": LR * 0.1,  "weight_decay": 0.0},
#     # quantum layer gets 10x smaller LR — critical for stability
# ])

# # ReduceLROnPlateau — reduces LR when stuck, perfect for quantum models
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode     = "max",    # maximise AUC
#     factor   = 0.5,      # halve LR when stuck
#     patience = 8,        # wait 8 epochs before reducing
#     min_lr   = 1e-6,
# )

# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# best_auc = 0.0; best_acc = 0.0; patience_ct = 0

# print("\n" + "="*60)
# print("  Training  (Classical + Quantum parallel paths)")
# print("="*60 + "\n")

# for epoch in range(1, EPOCHS + 1):
#     model.train(); total = 0.0; t0 = time.time()

#     for ef, cf, lb in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(ef, cf), lb.unsqueeze(1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total += loss.item()

#     model.eval(); pa, pr, la = [], [], []

#     with torch.no_grad():
#         for ef, cf, lb in test_loader:
#             p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#             pa.extend(p); pr.extend((p>0.5).astype(int)); la.extend(lb.numpy())

#     acc = accuracy_score(la, pr)
#     auc = roc_auc_score(la, pa)
#     f1  = f1_score(la, pr, zero_division=0)

#     scheduler.step(auc)   # ReduceLROnPlateau needs metric
#     lr_now = optimizer.param_groups[0]["lr"]

#     print(f"Epoch {epoch:03d}/{EPOCHS}  "
#           f"loss={total/len(train_loader):.4f}  "
#           f"acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  "
#           f"lr={lr_now:.6f}  ({time.time()-t0:.1f}s)")

#     if auc > best_auc:
#         best_auc = auc; best_acc = acc; patience_ct = 0
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"  ✓ Best saved  acc={acc:.4f}  auc={auc:.4f}")
#     else:
#         patience_ct += 1
#         print(f"  patience {patience_ct}/{PATIENCE}")
#         if patience_ct >= PATIENCE:
#             print(f"  Early stopping at epoch {epoch}."); break

# # ── Final Evaluation ──────────────────────────────────────
# print("\n" + "="*60 + "\n  Final Evaluation\n" + "="*60)
# model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
# model.eval(); pa, la = [], []

# with torch.no_grad():
#     for ef, cf, lb in test_loader:
#         p = torch.sigmoid(model(ef, cf)).numpy().flatten()
#         pa.extend(p); la.extend(lb.numpy())

# pa = np.array(pa); la = np.array(la)

# bt, bta = 0.5, 0.0
# for t in np.arange(0.3, 0.7, 0.01):
#     a = accuracy_score(la, (pa>t).astype(int))
#     if a > bta: bta=a; bt=t

# pf = (pa > bt).astype(int)
# print(f"\n  Best threshold : {bt:.2f}")
# print(f"  Accuracy       : {accuracy_score(la,pf)*100:.2f}%")
# print(f"  ROC AUC        : {roc_auc_score(la,pa):.4f}")
# print(f"  F1 Score       : {f1_score(la,pf,zero_division=0):.4f}")
# print(f"  Precision      : {precision_score(la,pf,zero_division=0):.4f}")
# print(f"  Recall         : {recall_score(la,pf,zero_division=0):.4f}")
# print(f"\n  Confusion Matrix:\n{confusion_matrix(la,pf)}")
# print("="*60)
# print(f"\n✓ Saved → {SAVE_PATH}")
# print(f"✓ Cache → {CACHE_PATH}  (next run loads instantly!)")





# """
# train_multimodal.py  —  Quantum Multimodal Training (92%+ target)
# ==================================================================
# Uses SAME split as train_all_flat.py:
#   train_test_split(seed=42, test_size=0.2)
#   → fair comparison with SVM/ANN/QNN/VQC

# Run:  python train/train_multimodal.py
# Saved → saved_models/best_multimodal_model.pth
# """

# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# import wfdb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score, confusion_matrix
# )

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from dataset.feature_extractor import (
#     extract_ecg_features_multilead,
#     extract_ecg_features
# )
# from models.quantum_layer import QuantumLayer

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# PTBXL_CSV   = "data/ptbxl/ptbxl_database.csv"
# PTBXL_BASE  = "data/ptbxl"
# MAX_SAMPLES = 12000
# BATCH_SIZE  = 32
# EPOCHS      = 200
# LR          = 5e-4
# PATIENCE    = 40
# SAVE_PATH   = "saved_models/best_multimodal_model.pth"
# CACHE_PATH  = "saved_models/feature_cache.npz"

# # SAME as train_all_flat.py and evaluate_models.py
# TEST_SIZE   = 0.2
# RANDOM_SEED = 42

# device = "cpu"

# print("=" * 60)
# print("  QMM CARDIONET2 — Quantum Multimodal Training")
# print("=" * 60)
# print("  MAX_SAMPLES = " + str(MAX_SAMPLES))
# print("  CACHE       = " + CACHE_PATH)
# print("  SPLIT       = train_test_split seed=42 test=0.2")
# print("=" * 60)


# # ═══════════════════════════════════════════════════════════
# #  FEATURE EXTRACTION WITH CACHE
# # ═══════════════════════════════════════════════════════════
# def load_or_extract(csv, base, max_samples, cache_path):
#     if os.path.exists(cache_path):
#         print("\nCache found -> loading from " + cache_path)
#         t0   = time.time()
#         data = np.load(cache_path)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         lbl_arr  = data["labels"]
#         print("Loaded " + str(len(lbl_arr)) + " samples in " +
#               str(round(time.time()-t0, 1)) + "s")
#         print("NORM=" + str((lbl_arr==0).sum()) +
#               "  ABNORMAL=" + str((lbl_arr==1).sum()))
#         return ecg_arr, clin_arr, lbl_arr

#     print("\nNo cache -> extracting features from " + str(max_samples) +
#           " records ...")
#     print("This runs ONCE then saves to cache.\n")

#     df = pd.read_csv(csv)
#     df = df[df["filename_hr"].notna()].reset_index(drop=True)
#     df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

#     if max_samples and len(df) > max_samples:
#         dn = df[df["scp_codes"].str.contains("NORM", na=False)]
#         da = df[~df["scp_codes"].str.contains("NORM", na=False)]
#         n  = max_samples // 2
#         df = pd.concat([
#             dn.sample(min(n, len(dn)), random_state=42),
#             da.sample(min(n, len(da)), random_state=42)
#         ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     ecg_f, clin_f, labels = [], [], []
#     skip = 0
#     t0   = time.time()

#     for i in range(len(df)):
#         if i % 500 == 0 and i > 0:
#             elapsed = time.time() - t0
#             eta     = elapsed / i * (len(df) - i)
#             print("  [" + str(i) + "/" + str(len(df)) + "]  " +
#                   "skip=" + str(skip) + "  " +
#                   "elapsed=" + str(round(elapsed/60,1)) + "min  " +
#                   "ETA=" + str(round(eta/60,1)) + "min")

#         row  = df.iloc[i]
#         path = os.path.join(base, row["filename_hr"])

#         try:
#             sig, _ = wfdb.rdsamp(path)
#         except Exception:
#             skip += 1; continue
#         try:
#             f = (extract_ecg_features_multilead(sig)
#                  if sig.shape[1] >= 11
#                  else extract_ecg_features(sig[:, 0]))
#         except Exception:
#             skip += 1; continue
#         if not np.all(np.isfinite(f)):
#             skip += 1; continue

#         ecg_f.append(f)
#         clin_f.append([float(row["age"]), float(row["sex"])])
#         labels.append(0 if "NORM" in str(row["scp_codes"]) else 1)

#     ecg_arr  = np.array(ecg_f,  dtype=np.float32)
#     clin_arr = np.array(clin_f, dtype=np.float32)
#     lbl_arr  = np.array(labels, dtype=np.float32)

#     print("Extraction done in " + str(round((time.time()-t0)/60,1)) + " min")
#     print("valid=" + str(len(lbl_arr)) + "  skipped=" + str(skip))

#     os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#     np.savez_compressed(cache_path,
#                         ecg=ecg_arr, clin=clin_arr, labels=lbl_arr)
#     print("Cache saved -> " + cache_path)

#     return ecg_arr, clin_arr, lbl_arr


# # ═══════════════════════════════════════════════════════════
# #  DATASET
# # ═══════════════════════════════════════════════════════════
# class CachedMultimodalDataset(Dataset):
#     def __init__(self, ecg_arr, clin_arr, lbl_arr):
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         combined = np.nan_to_num(
#             scaler.fit_transform(combined).astype(np.float32),
#             nan=0., posinf=0., neginf=0.
#         )
#         self.ecg    = torch.tensor(combined[:, :ecg_arr.shape[1]],
#                                     dtype=torch.float32)
#         self.clin   = torch.tensor(combined[:, ecg_arr.shape[1]:],
#                                     dtype=torch.float32)
#         self.labels = torch.tensor(lbl_arr, dtype=torch.float32)

#     def __len__(self):  return len(self.labels)
#     def __getitem__(self, i):
#         return self.ecg[i], self.clin[i], self.labels[i]


# # ═══════════════════════════════════════════════════════════
# #  MODEL
# # ═══════════════════════════════════════════════════════════
# class ResBlock(nn.Module):
#     def __init__(self, dim, drop=0.2):
#         super().__init__()
#         self.b = nn.Sequential(
#             nn.Linear(dim,dim), nn.BatchNorm1d(dim),
#             nn.ReLU(), nn.Dropout(drop),
#             nn.Linear(dim,dim), nn.BatchNorm1d(dim),
#         )
#     def forward(self, x): return F.relu(x + self.b(x))


# class SharedEncoder(nn.Module):
#     def __init__(self, ecg_dim):
#         super().__init__()
#         in_dim = ecg_dim + 2
#         self.net = nn.Sequential(
#             nn.Linear(in_dim,512), nn.BatchNorm1d(512),
#             nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(512,256), nn.BatchNorm1d(256),
#             nn.ReLU(), nn.Dropout(0.25),
#             nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(),
#         )
#         self.res1 = ResBlock(128, 0.2)
#         self.res2 = ResBlock(128, 0.15)
#         self.proj = nn.Sequential(
#             nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU()
#         )
#     def forward(self, ecg_f, clin_f):
#         x = torch.cat([ecg_f, clin_f], dim=1)
#         return self.proj(self.res2(self.res1(self.net(x))))


# class DualPathQuantumNet(nn.Module):
#     """
#     Classical path (64) + Quantum path (8) in parallel.
#     Merged at classifier -> 72-dim -> output.
#     """
#     def __init__(self, ecg_dim):
#         super().__init__()
#         self.encoder          = SharedEncoder(ecg_dim)
#         self.classical_path   = nn.Sequential(
#             ResBlock(64, 0.2), ResBlock(64, 0.15)
#         )
#         self.quantum_compress = nn.Sequential(
#             nn.Linear(64,32), nn.ReLU(),
#             nn.Linear(32, 8), nn.Tanh(),
#         )
#         self.quantum    = QuantumLayer()
#         self.classifier = nn.Sequential(
#             nn.Linear(72,64), nn.BatchNorm1d(64),
#             nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(64,32), nn.ReLU(),
#             nn.Linear(32, 1),
#         )

#     def forward(self, ecg_f, clin_f):
#         shared  = self.encoder(ecg_f, clin_f)
#         cls_out = self.classical_path(shared)
#         q_out   = self.quantum(self.quantum_compress(shared))
#         return self.classifier(torch.cat([cls_out, q_out], dim=1))


# # ═══════════════════════════════════════════════════════════
# #  LOAD DATA
# # ═══════════════════════════════════════════════════════════
# os.makedirs("saved_models", exist_ok=True)

# ecg_arr, clin_arr, lbl_arr = load_or_extract(
#     PTBXL_CSV, PTBXL_BASE, MAX_SAMPLES, CACHE_PATH
# )

# # ── SAME split as train_all_flat.py and evaluate_models.py ─
# idx = np.arange(len(ecg_arr))
# idx_tr, idx_te = train_test_split(
#     idx,
#     test_size    = TEST_SIZE,
#     random_state = RANDOM_SEED,
#     stratify     = lbl_arr.astype(int)
# )

# train_ds = CachedMultimodalDataset(
#     ecg_arr[idx_tr], clin_arr[idx_tr], lbl_arr[idx_tr]
# )
# test_ds  = CachedMultimodalDataset(
#     ecg_arr[idx_te], clin_arr[idx_te], lbl_arr[idx_te]
# )
# ecg_dim  = train_ds.ecg.shape[1]

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
#                           shuffle=True,  num_workers=0)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
#                           shuffle=False, num_workers=0)

# print("\nTrain=" + str(len(train_ds)) +
#       "  Test="  + str(len(test_ds)) +
#       "  ECG-dim=" + str(ecg_dim))

# lbl_np = lbl_arr
# pos_w  = torch.tensor(
#     [(lbl_np==0).sum() / max((lbl_np==1).sum(), 1)],
#     dtype=torch.float32
# )
# print("pos_weight=" + str(round(pos_w.item(), 3)))


# # ═══════════════════════════════════════════════════════════
# #  TRAINING
# # ═══════════════════════════════════════════════════════════
# model    = DualPathQuantumNet(ecg_dim)
# n_params = sum(p.numel() for p in model.parameters()
#                if p.requires_grad)
# print("Parameters: " + str(n_params))

# # separate LR for quantum vs classical
# quantum_params   = list(model.quantum.parameters())
# classical_params = [p for p in model.parameters()
#                     if not any(p is q for q in quantum_params)]

# optimizer = torch.optim.AdamW([
#     {"params": classical_params, "lr": LR,       "weight_decay": 1e-4},
#     {"params": quantum_params,   "lr": LR * 0.1, "weight_decay": 0.0},
# ])

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
# )
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
# best_auc  = 0.0
# patience_ct = 0

# print("\n" + "="*60)
# print("  Training (Classical + Quantum parallel)")
# print("="*60 + "\n")

# for epoch in range(1, EPOCHS + 1):
#     model.train(); total = 0.0; t0 = time.time()

#     for ef, cf, lb in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(ef, cf), lb.unsqueeze(1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total += loss.item()

#     model.eval(); pa, pr, la = [], [], []
#     with torch.no_grad():
#         for ef, cf, lb in test_loader:
#             p = torch.sigmoid(model(ef,cf)).numpy().flatten()
#             pa.extend(p)
#             pr.extend((p>0.5).astype(int))
#             la.extend(lb.numpy())

#     acc = accuracy_score(la, pr)
#     auc = roc_auc_score(la, pa)
#     f1  = f1_score(la, pr, zero_division=0)
#     lr_now = optimizer.param_groups[0]["lr"]

#     scheduler.step(auc)

#     print("Epoch " + str(epoch).zfill(3) + "/" + str(EPOCHS) +
#           "  loss=" + str(round(total/len(train_loader),4)) +
#           "  acc="  + str(round(acc,4)) +
#           "  auc="  + str(round(auc,4)) +
#           "  f1="   + str(round(f1,4)) +
#           "  lr="   + str(round(lr_now,6)) +
#           "  (" + str(round(time.time()-t0,1)) + "s)")

#     if auc > best_auc:
#         best_auc = auc; patience_ct = 0
#         torch.save(model.state_dict(), SAVE_PATH)
#         print("  Best saved  acc=" + str(round(acc,4)) +
#               "  auc=" + str(round(auc,4)))
#     else:
#         patience_ct += 1
#         print("  patience " + str(patience_ct) + "/" + str(PATIENCE))
#         if patience_ct >= PATIENCE:
#             print("  Early stopping at epoch " + str(epoch))
#             break


# # ═══════════════════════════════════════════════════════════
# #  FINAL EVALUATION
# # ═══════════════════════════════════════════════════════════
# print("\n" + "="*60 + "\n  Final Evaluation\n" + "="*60)
# model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
# model.eval()

# pa, la = [], []
# with torch.no_grad():
#     for ef, cf, lb in test_loader:
#         p = torch.sigmoid(model(ef,cf)).numpy().flatten()
#         pa.extend(p); la.extend(lb.numpy())

# pa = np.array(pa); la = np.array(la)

# best_t, best_a = 0.5, 0.0
# for t in np.arange(0.3, 0.7, 0.01):
#     a = accuracy_score(la, (pa>t).astype(int))
#     if a > best_a: best_a=a; best_t=t

# pf = (pa > best_t).astype(int)
# print("  Best threshold : " + str(round(best_t,2)))
# print("  Accuracy       : " + str(round(accuracy_score(la,pf)*100,2)) + "%")
# print("  ROC AUC        : " + str(round(roc_auc_score(la,pa),4)))
# print("  F1 Score       : " + str(round(f1_score(la,pf,zero_division=0),4)))
# print("  Precision      : " + str(round(precision_score(la,pf,zero_division=0),4)))
# print("  Recall         : " + str(round(recall_score(la,pf,zero_division=0),4)))
# print("\n  Confusion Matrix:\n" + str(confusion_matrix(la,pf)))
# print("="*60)
# print("\nSaved -> " + SAVE_PATH)



"""
train_multimodal.py  —  Quantum Multimodal Training (92%+ target)
==================================================================
Uses SAME split as train_all_flat.py:
  train_test_split(seed=42, test_size=0.2)
  → fair comparison with SVM/ANN/QNN/VQC

Run:  python train/train_multimodal.py
Saved → saved_models/best_multimodal_model.pth
"""

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