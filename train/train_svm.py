# """
# train_svm.py  —  Complete Final Version
# ========================================
# Just run:  python train/train_svm.py
# Model saved → saved_models/svm_model.pkl
# """

# import os
# import joblib
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing   import StandardScaler
# from sklearn.metrics          import (
#     accuracy_score, classification_report, confusion_matrix
# )
# from models.classical.svm_model  import create_svm
# from dataset.dataset_builder     import build_flat_dataset


# # ── Dataset ──────────────────────────────────────────────
# X, y = build_flat_dataset(multilead=True)

# # ── Split ─────────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ── Scale ─────────────────────────────────────────────────
# scaler  = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)

# # ── Train ─────────────────────────────────────────────────
# model = create_svm()
# print("\nTraining SVM …")
# model.fit(X_train, y_train)

# # ── Evaluate ──────────────────────────────────────────────
# pred = model.predict(X_test)
# print("\nSVM Accuracy :", accuracy_score(y_test, pred))
# print("\nConfusion Matrix:\n",      confusion_matrix(y_test, pred))
# print("\nClassification Report:\n", classification_report(y_test, pred))

# # ── Save ──────────────────────────────────────────────────
# os.makedirs("saved_models", exist_ok=True)
# joblib.dump(model,  "saved_models/svm_model.pkl")
# joblib.dump(scaler, "saved_models/svm_scaler.pkl")
# print("\n✓  Saved → saved_models/svm_model.pkl")








"""
train_svm.py  —  Fair Comparison Version
==========================================
IMPORTANT FOR RESEARCH PAPER:
  Uses the SAME cached dataset as train_multimodal.py
  → fair apple-to-apple comparison between all models
  → all models trained on identical 12000 samples

Run:  python train/train_svm.py
Saved → saved_models/svm_model.pkl
"""

import os, sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────
CACHE_PATH = "saved_models/feature_cache.npz"   # same as multimodal
SAVE_PATH  = "saved_models/svm_model.pkl"
SCALER_PATH= "saved_models/svm_scaler.pkl"

print("=" * 55)
print("  SVM Training  (same data as Multimodal)")
print("=" * 55)


# ── Load from cache (same data multimodal used) ───────────
if os.path.exists(CACHE_PATH):
    print(f"\nLoading from cache: {CACHE_PATH}")
    data     = np.load(CACHE_PATH)
    ecg_arr  = data["ecg"]
    clin_arr = data["clin"]
    lbl_arr  = data["labels"].astype(int)
    # combine ecg + clinical features
    X = np.concatenate([ecg_arr, clin_arr], axis=1)
    y = lbl_arr
    print(f"Loaded {len(y)} samples  "
          f"NORM={(y==0).sum()}  ABNORMAL={(y==1).sum()}")
else:
    # fallback: build from scratch
    print(f"Cache not found — building dataset from scratch …")
    from dataset.dataset_builder import build_flat_dataset
    X, y = build_flat_dataset(multilead=True)

print(f"Feature dim: {X.shape[1]}")


# ── Train/Test split (same seed as multimodal) ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.2,
    random_state= 42,
    stratify    = y
)
print(f"\nTrain={len(X_train)}  Test={len(X_test)}")


# ── Scale ─────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
X_train = np.nan_to_num(X_train, nan=0., posinf=0., neginf=0.)
X_test  = np.nan_to_num(X_test,  nan=0., posinf=0., neginf=0.)


# ── Train SVM ─────────────────────────────────────────────
from sklearn.svm import SVC

model = SVC(
    kernel      = "rbf",
    C           = 10,
    gamma       = "scale",
    probability = True,
    random_state= 42
)

print("\nTraining SVM …")
model.fit(X_train, y_train)


# ── Evaluate ──────────────────────────────────────────────
pred  = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, probs)

print(f"\nSVM Results:")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  ROC AUC  : {auc:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test, pred, target_names=['Normal','Abnormal'])}")


# ── Save ──────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model,  SAVE_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✓ SVM    saved → {SAVE_PATH}")
print(f"✓ Scaler saved → {SCALER_PATH}")