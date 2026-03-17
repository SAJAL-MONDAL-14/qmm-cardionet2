import os
import wfdb
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MultimodalDataset(Dataset):

    def __init__(
        self,
        ptbxl_csv,
        ptbxl_base="data/ptbxl",
        max_len=1000,
        max_samples=None,
        cache=True
    ):

        self.ptbxl_base = ptbxl_base
        self.max_len = max_len
        self.cache = cache

        # =========================
        # Load PTBXL metadata
        # =========================

        self.ecg = pd.read_csv(ptbxl_csv)

        self.ecg = self.ecg[self.ecg["filename_hr"].notna()].reset_index(drop=True)

        self.ecg = self.ecg[
            self.ecg["age"].notna() &
            self.ecg["sex"].notna()
        ].reset_index(drop=True)

        # limit dataset size (speed training)
        if max_samples is not None:
            self.ecg = self.ecg.iloc[:max_samples]

        # =========================
        # Clinical Features
        # =========================

        clinical = self.ecg[["age", "sex"]].astype(np.float32)

        scaler = StandardScaler()
        self.clinical = scaler.fit_transform(clinical)

        # =========================
        # Labels
        # =========================

        self.labels = []

        for scp in self.ecg["scp_codes"]:

            label = 0 if "NORM" in scp else 1
            self.labels.append(label)

        self.labels = np.array(self.labels)

        # =========================
        # Class imbalance weight
        # =========================

        pos = np.sum(self.labels == 1)
        neg = np.sum(self.labels == 0)

        self.pos_weight = neg / pos

        # =========================
        # ECG cache (speed boost)
        # =========================

        self.ecg_cache = {}

    def __len__(self):
        return len(self.ecg)

    def load_ecg(self, idx):

        if self.cache and idx in self.ecg_cache:
            return self.ecg_cache[idx]

        record_path = os.path.join(
            self.ptbxl_base,
            self.ecg.iloc[idx]["filename_hr"]
        )

        signal, _ = wfdb.rdsamp(record_path)

        signal = signal[:, 0]

        # trim or pad
        if len(signal) > self.max_len:
            signal = signal[:self.max_len]
        else:
            pad = self.max_len - len(signal)
            signal = np.pad(signal, (0, pad))

        # =========================
        # Better ECG normalization
        # =========================

        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # remove extreme noise spikes
        signal = np.clip(signal, -5, 5)

        signal = torch.tensor(signal, dtype=torch.float32)

        if self.cache:
            self.ecg_cache[idx] = signal

        return signal

    def __getitem__(self, idx):

        clinical_x = torch.tensor(
            self.clinical[idx],
            dtype=torch.float32
        )

        signal = self.load_ecg(idx)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return clinical_x, signal, label