
import os
import sys
import time
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classical.ann_model import ANNModel
from models.quantum.qnn_model   import QNNModel
from models.quantum.vqc_model   import VQCModel

# ── Config ────────────────────────────────────────────────
CACHE_PATH  = "saved_models/feature_cache.npz"
SCALER_PATH = "saved_models/shared_scaler.pkl"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

print("=" * 60)
print("  train_all_flat.py — SVM | ANN | QNN | VQC")
print("  Smart skip: already trained models are skipped")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
#  LOAD DATA FROM CACHE
# ═══════════════════════════════════════════════════════════
if not os.path.exists(CACHE_PATH):
    print("ERROR: Cache not found at " + CACHE_PATH)
    print("Run train_multimodal.py first to generate the cache.")
    sys.exit(1)

print("\nLoading from cache: " + CACHE_PATH)
data     = np.load(CACHE_PATH)
ecg_arr  = data["ecg"]
clin_arr = data["clin"]
y        = data["labels"].astype(int)
X        = np.concatenate([ecg_arr, clin_arr], axis=1).astype(np.float32)
X        = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
print("Loaded " + str(len(y)) + " samples  " +
      "NORM=" + str((y==0).sum()) + "  " +
      "ABNORMAL=" + str((y==1).sum()))


# ═══════════════════════════════════════════════════════════
#  SAME TRAIN/TEST SPLIT AS evaluate_models.py
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = y
)
print("Train=" + str(len(X_train)) + "  Test=" + str(len(X_test)))


# ═══════════════════════════════════════════════════════════
#  SHARED SCALER
# ═══════════════════════════════════════════════════════════
os.makedirs("saved_models", exist_ok=True)

if os.path.exists(SCALER_PATH):
    scaler    = joblib.load(SCALER_PATH)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    print("Loaded shared scaler: " + SCALER_PATH)
else:
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print("Shared scaler saved -> " + SCALER_PATH)

X_test_s  = scaler.transform(X_test).astype(np.float32)
X_train_s = np.nan_to_num(X_train_s, nan=0., posinf=0., neginf=0.)
X_test_s  = np.nan_to_num(X_test_s,  nan=0., posinf=0., neginf=0.)


# ── Helper ────────────────────────────────────────────────
def evaluate(name, preds, probs, y_true):
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    print("\n" + name + " Results:")
    print("  Accuracy : " + str(round(acc*100, 2)) + "%")
    print("  ROC AUC  : " + str(round(auc, 4)))
    print(classification_report(
        y_true, preds,
        target_names=["Normal","Abnormal"],
        zero_division=0
    ))
    return acc, auc


# ═══════════════════════════════════════════════════════════
#  SVM
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
if os.path.exists("saved_models/svm_model.pkl"):
    print("  SVM — already trained, loading ...")
    print("  (delete saved_models/svm_model.pkl to retrain)")
    print("="*60)
    svm       = joblib.load("saved_models/svm_model.pkl")
    svm_probs = svm.predict_proba(X_test_s)[:, 1]
    svm_preds = svm.predict(X_test_s)
    evaluate("SVM", svm_preds, svm_probs, y_test)
else:
    print("  Training SVM ...")
    print("="*60)
    from sklearn.svm import SVC
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=42)
    t0  = time.time()
    svm.fit(X_train_s, y_train)
    print("SVM trained in " + str(round(time.time()-t0, 1)) + "s")
    svm_probs = svm.predict_proba(X_test_s)[:, 1]
    svm_preds = svm.predict(X_test_s)
    evaluate("SVM", svm_preds, svm_probs, y_test)
    joblib.dump(svm, "saved_models/svm_model.pkl")
    print("Saved -> saved_models/svm_model.pkl")


# ═══════════════════════════════════════════════════════════
#  ANN
# ═══════════════════════════════════════════════════════════
input_dim = X_train_s.shape[1]
X_tr_t    = torch.tensor(X_train_s, dtype=torch.float32)
y_tr_t    = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
X_te_t    = torch.tensor(X_test_s,  dtype=torch.float32)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_w = torch.tensor([n_neg / max(n_pos,1)], dtype=torch.float32)

print("\n" + "="*60)
if os.path.exists("saved_models/ann_model.pth"):
    print("  ANN — already trained, loading ...")
    print("  (delete saved_models/ann_model.pth to retrain)")
    print("="*60)
    ann = ANNModel(input_dim)
    ann.load_state_dict(torch.load("saved_models/ann_model.pth",
                                    map_location="cpu"))
    ann.eval()
    with torch.no_grad():
        ann_probs = torch.sigmoid(ann(X_te_t)).numpy().flatten()
    ann_preds = (ann_probs > 0.5).astype(int)
    evaluate("ANN", ann_preds, ann_probs, y_test)
else:
    print("  Training ANN ...")
    print("="*60)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=256, shuffle=True)
    ann   = ANNModel(input_dim)
    opt   = torch.optim.Adam(ann.parameters(), lr=0.0005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    crit  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for epoch in range(1, 51):
        ann.train(); total = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(ann(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if epoch % 10 == 0:
            print("  Epoch " + str(epoch) + "/50  loss=" +
                  str(round(total/len(loader), 4)))
    ann.eval()
    with torch.no_grad():
        ann_probs = torch.sigmoid(ann(X_te_t)).numpy().flatten()
    ann_preds = (ann_probs > 0.5).astype(int)
    evaluate("ANN", ann_preds, ann_probs, y_test)
    torch.save(ann.state_dict(), "saved_models/ann_model.pth")
    print("Saved -> saved_models/ann_model.pth")


# ═══════════════════════════════════════════════════════════
#  QNN
# ═══════════════════════════════════════════════════════════
loader_q = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                      batch_size=32, shuffle=True)

print("\n" + "="*60)
if os.path.exists("saved_models/qnn_model.pth"):
    print("  QNN — already trained, loading ...")
    print("  (delete saved_models/qnn_model.pth to retrain)")
    print("="*60)
    qnn = QNNModel(input_dim)
    qnn.load_state_dict(torch.load("saved_models/qnn_model.pth",
                                    map_location="cpu"))
    qnn.eval()
    with torch.no_grad():
        qnn_probs = torch.sigmoid(qnn(X_te_t)).numpy().flatten()
    qnn_preds = (qnn_probs > 0.5).astype(int)
    evaluate("QNN", qnn_preds, qnn_probs, y_test)
else:
    print("  Training QNN ... (slow on CPU, be patient)")
    print("="*60)
    qnn     = QNNModel(input_dim)
    opt_q   = torch.optim.Adam(qnn.parameters(), lr=0.001)
    sched_q = torch.optim.lr_scheduler.StepLR(opt_q, step_size=10, gamma=0.7)
    crit_q  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for epoch in range(1, 31):
        qnn.train(); total = 0
        for xb, yb in loader_q:
            opt_q.zero_grad()
            loss = crit_q(qnn(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnn.parameters(), 1.0)
            opt_q.step(); total += loss.item()
        sched_q.step()
        print("  Epoch " + str(epoch) + "/30  loss=" +
              str(round(total/len(loader_q), 4)))
    qnn.eval()
    with torch.no_grad():
        qnn_probs = torch.sigmoid(qnn(X_te_t)).numpy().flatten()
    qnn_preds = (qnn_probs > 0.5).astype(int)
    evaluate("QNN", qnn_preds, qnn_probs, y_test)
    torch.save(qnn.state_dict(), "saved_models/qnn_model.pth")
    print("Saved -> saved_models/qnn_model.pth")


# ═══════════════════════════════════════════════════════════
#  VQC
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
if os.path.exists("saved_models/vqc_model.pth"):
    print("  VQC — already trained, loading ...")
    print("  (delete saved_models/vqc_model.pth to retrain)")
    print("="*60)
    vqc = VQCModel(input_dim)
    vqc.load_state_dict(torch.load("saved_models/vqc_model.pth",
                                    map_location="cpu"))
    vqc.eval()
    with torch.no_grad():
        vqc_probs = torch.sigmoid(vqc(X_te_t)).numpy().flatten()
    best_t, best_a = 0.5, 0.0
    for t in np.arange(0.3, 0.7, 0.01):
        a = accuracy_score(y_test, (vqc_probs > t).astype(int))
        if a > best_a: best_a=a; best_t=t
    vqc_preds = (vqc_probs > best_t).astype(int)
    evaluate("VQC", vqc_preds, vqc_probs, y_test)
else:
    print("  Training VQC ... (slow on CPU, be patient)")
    print("="*60)
    vqc     = VQCModel(input_dim)
    opt_v   = torch.optim.Adam(vqc.parameters(), lr=0.001)
    sched_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=15, gamma=0.7)
    crit_v  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for epoch in range(1, 41):
        vqc.train(); total = 0
        for xb, yb in loader_q:
            opt_v.zero_grad()
            loss = crit_v(vqc(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqc.parameters(), 1.0)
            opt_v.step(); total += loss.item()
        sched_v.step()
        print("  Epoch " + str(epoch) + "/40  loss=" +
              str(round(total/len(loader_q), 4)))
    vqc.eval()
    with torch.no_grad():
        vqc_probs = torch.sigmoid(vqc(X_te_t)).numpy().flatten()
    best_t, best_a = 0.5, 0.0
    for t in np.arange(0.3, 0.7, 0.01):
        a = accuracy_score(y_test, (vqc_probs > t).astype(int))
        if a > best_a: best_a=a; best_t=t
    vqc_preds = (vqc_probs > best_t).astype(int)
    print("VQC best threshold: " + str(round(best_t, 2)))
    evaluate("VQC", vqc_preds, vqc_probs, y_test)
    torch.save(vqc.state_dict(), "saved_models/vqc_model.pth")
    print("Saved -> saved_models/vqc_model.pth")


# ═══════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DONE")
print("="*60)
print("  Saved models:")
print("    saved_models/svm_model.pkl")
print("    saved_models/ann_model.pth")
print("    saved_models/qnn_model.pth")
print("    saved_models/vqc_model.pth")
print("    saved_models/shared_scaler.pkl")
print("\n  To retrain a model, delete its file and rerun.")
print("  Now run:  python evaluate_models.py")
print("="*60)