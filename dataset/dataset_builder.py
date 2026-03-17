"""
dataset_builder.py  —  Shared dataset builder for all train scripts
====================================================================
Usage:
    from dataset.dataset_builder import build_flat_dataset
    X, y = build_flat_dataset(multilead=True)
"""

import os
import numpy as np
import pandas as pd
import wfdb
from dataset.feature_extractor import extract_ecg_features_multilead, extract_ecg_features

PTBXL_CSV  = "data/ptbxl/ptbxl_database.csv"
PTBXL_BASE = "data/ptbxl"


def build_flat_dataset(
    ptbxl_csv   = PTBXL_CSV,
    ptbxl_base  = PTBXL_BASE,
    max_samples = None,
    multilead   = True,
    verbose     = True
):
    """
    Returns X (n_samples, n_features) and y (n_samples,).
    multilead=True  →  5 leads × 65 features = 325 features
    multilead=False →  1 lead  × 65 features =  65 features
    """
    df = pd.read_csv(ptbxl_csv)
    df = df[df["filename_hr"].notna()].reset_index(drop=True)
    df = df[df["age"].notna() & df["sex"].notna()].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    X, y    = [], []
    skipped = 0

    if verbose:
        print(f"Building dataset from {len(df)} records "
              f"({'multi-lead 5×65=325 feats' if multilead else 'single-lead 65 feats'}) …")

    for i in range(len(df)):
        if verbose and i % 500 == 0 and i > 0:
            print(f"  {i}/{len(df)}  (skipped={skipped})")

        row  = df.iloc[i]
        path = os.path.join(ptbxl_base, row["filename_hr"])

        try:
            signal, _ = wfdb.rdsamp(path)
        except Exception:
            skipped += 1
            continue

        try:
            if multilead and signal.shape[1] >= 11:
                ecg_feats = extract_ecg_features_multilead(signal)
            else:
                ecg_feats = extract_ecg_features(signal[:, 0])
        except Exception:
            skipped += 1
            continue

        fvec  = np.concatenate([[float(row["age"]), float(row["sex"])], ecg_feats])
        label = 0 if "NORM" in str(row["scp_codes"]) else 1

        X.append(fvec)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # remove any NaN/Inf rows
    valid = np.all(np.isfinite(X), axis=1)
    X, y  = X[valid], y[valid]

    if verbose:
        print(f"\nDataset ready → shape={X.shape}  skipped={skipped}")
        print(f"Class dist → NORM={(y==0).sum()}  OTHER={(y==1).sum()}")

    return X, y