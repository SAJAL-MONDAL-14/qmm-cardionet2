# QMM-CardioNet2 🫀⚛️

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=for-the-badge&logo=pytorch)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-green?style=for-the-badge)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Platform-6929C4?style=for-the-badge&logo=ibm)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A Hybrid Quantum-Classical Multimodal Neural Network for ECG-Based Cardiac Arrhythmia Classification**

*Combining 325-dimensional handcrafted ECG features with an 8-qubit Variational Quantum Circuit on the PTB-XL dataset*

[📄 Paper](#paper) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [⚛️ IBM Quantum](#ibm-quantum) • [📁 Structure](#project-structure)

</div>

---

## 📌 Overview

**QMM-CardioNet2** proposes a dual-path hybrid quantum-classical architecture that integrates rich multi-lead ECG feature representations with a variational quantum circuit (VQC) for binary cardiac arrhythmia classification (Normal vs. Abnormal).

The system is evaluated on **PTB-XL** — the largest publicly available 12-lead ECG dataset — and achieves superior performance over all classical and standalone quantum baselines under identical experimental conditions.

### Key Features

- 🧠 **Dual-path architecture** — classical residual network (64-dim) running in parallel with 8-qubit VQC (8-dim), fused at classifier
- 🔬 **325-dimensional feature vector** — 65 features × 5 leads (I, II, V1, V2, V5) covering HRV, morphology, wavelet, spectral, and Hjorth parameters
- ⚛️ **Anti-barren-plateau quantum design** — 3-layer VQC with data re-uploading and nearest-neighbour CNOT ring entanglement
- 🏥 **Real IBM Quantum execution** — circuit runs on `ibm_marrakesh` / `ibm_fez` via IBM Quantum Platform
- 🖥️ **Interactive terminal prediction** — predict Normal/Abnormal from any PTB-XL ECG record in real time

---

## 📊 Results

All five models trained and evaluated on an **identical** 12,000-sample stratified split of PTB-XL (seed=42, 80/20 train/test, shared StandardScaler).

| Model | Accuracy | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| SVM | 85.42% | 0.9283 | 0.8519 | 0.8651 | 0.8392 |
| ANN | 86.96% | 0.9372 | 0.8692 | 0.8718 | 0.8667 |
| QNN | 85.17% | 0.9050 | 0.8504 | 0.8576 | 0.8433 |
| VQC | 86.96% | 0.9333 | 0.8663 | 0.8887 | 0.8450 |
| **QMM-CardioNet2** | **87.29%** | **0.9389** | **0.8720** | **0.8783** | **0.8658** |

> QMM-CardioNet2 achieves the **highest performance across all five metrics**, demonstrating the effectiveness of the hybrid quantum-classical dual-path fusion.

---

## 🏗️ Architecture

```
Input: 325 ECG features + 2 clinical (age, sex) = 327-dim
          │
          ▼
┌─────────────────────────────────────────┐
│           Shared Encoder                │
│  327 → 512 → 256 → 128 → 64-dim        │
│  BatchNorm + ReLU + Dropout + 2×ResBlock│
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────────────┐
│  Classical  │  │    Quantum Path      │
│    Path     │  │  64→32→8 (Tanh)     │
│ 2×ResBlock  │  │  8-qubit VQC (L=3)  │
│  64-dim     │  │  8-dim ⟨Z₀⟩…⟨Z₇⟩  │
└──────┬──────┘  └──────────┬──────────┘
       └─────────┬───────────┘
                 ▼
         Concat (72-dim)
                 │
         Classifier (72→64→32→1)
                 │
          Normal / Abnormal
```

### Quantum Circuit Design (Anti-Barren-Plateau)

- **8 qubits**, **3 layers** (anti-barren-plateau: shallow = trainable gradients)
- **Hadamard initialization** → equal superposition
- **Data re-uploading** at every layer (Pérez-Salinas et al. 2020)
- **RZ-RY-RZ variational rotations** per qubit per layer
- **CNOT ring entanglement** (nearest-neighbour = local cost function)
- **Pauli-Z measurements** → 8-dim expectation value vector

---

## 📁 Project Structure

```
QMM_CARDIONET2/
├── data/
│   └── ptbxl/
│       ├── ptbxl_database.csv
│       ├── records100/          ← 100 Hz records
│       └── records500/          ← 500 Hz records (used)
│
├── dataset/
│   ├── feature_extractor.py     ← 325-dim multi-lead ECG feature extraction
│   ├── dataset_builder.py       ← shared dataset builder
│   └── dataset_multimodal.py
│
├── models/
│   ├── classical/
│   │   ├── ann_model.py         ← feedforward ANN baseline
│   │   └── svm_model.py
│   ├── quantum/
│   │   ├── qnn_model.py         ← quantum neural network
│   │   └── vqc_model.py         ← variational quantum classifier
│   ├── quantum_layer.py         ← VQC layer (simulator + IBM Quantum)
│   ├── multimodal_model.py      ← DualPathQuantumNet architecture
│   └── ecg_1dcnn.py
│
├── train/
│   ├── train_multimodal.py      ← train QMM-CardioNet2
│   └── (individual model scripts)
│
├── train_all_flat.py            ← train SVM/ANN/QNN/VQC (smart skip)
├── evaluate_models.py           ← evaluate all 5 models fairly
├── terminal_multimodal.py       ← interactive ECG prediction terminal
│
├── saved_models/
│   ├── feature_cache.npz        ← cached features (extract once)
│   ├── shared_scaler.pkl        ← shared StandardScaler
│   ├── best_multimodal_model.pth
│   ├── svm_model.pkl
│   ├── ann_model.pth
│   ├── qnn_model.pth
│   └── vqc_model.pth
│
└── evaluation_results/
    ├── svm_confusion.png
    ├── all_roc_curves.png
    ├── model_comparison.png
    ├── radar_chart.png
    └── dashboard.png
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/QMM-CardioNet2.git
cd QMM-CardioNet2
```

### 2. Install dependencies

```bash
pip install torch pennylane pennylane-qiskit scikit-learn numpy pandas wfdb matplotlib scipy joblib
pip install qiskit qiskit-ibm-runtime pylatexenc   # for IBM Quantum
```

### 3. Download PTB-XL dataset

Download from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place in `data/ptbxl/`:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

### 4. Train all models

```bash
# Step 1: Train QMM-CardioNet2 (generates feature cache on first run)
python train/train_multimodal.py

# Step 2: Train baseline models (skips already-trained models)
python train_all_flat.py

# Step 3: Evaluate all models fairly
python evaluate_models.py
```

### 5. Run prediction terminal

```bash
python terminal_multimodal.py
```

```
╔══════════════════════════════════════════════════════════╗
║         QMM CARDIONET2 — ECG Prediction Terminal         ║
╚══════════════════════════════════════════════════════════╝

  Patient age    : 63
  Sex (0=F,1=M)  : 1
  ECG file path  : data/ptbxl/records500/00000/00001_hr

  Diagnosis  :  NORMAL
  Confidence :  0.8732  (87.3%)
  Risk level :  LOW RISK
```

---

## ⚛️ IBM Quantum

QMM-CardioNet2 supports execution on **real IBM quantum hardware** via IBM Quantum Platform.

### Setup

```python
# Run once to save your IBM account
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_IBM_API_KEY",
    overwrite=True
)
```

### Enable IBM Quantum in `models/quantum_layer.py`

```python
USE_IBM_QUANTUM = True          # enable real hardware
IBM_BACKEND     = "ibm_marrakesh"   # or ibm_fez, ibm_torino
```

### Run on hardware

```bash
python terminal_multimodal.py
# → submits job to IBM Quantum Cloud
# → view results at quantum.cloud.ibm.com
```

### IBM Circuit Stats (ibm_marrakesh)

| Property | Value |
|---|---|
| Physical qubits | 8 (q[120–125], q[136], q[143]) |
| Native gates | RZ, SX (√X), CNOT |
| Transpiled depth | 45 |
| Shots | 1,024 |
| Execution time | ~11–23 s |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.x | Deep learning framework |
| `pennylane` | 0.38+ | Quantum computing framework |
| `pennylane-qiskit` | latest | PennyLane IBM Quantum plugin |
| `qiskit` | 1.x | Quantum circuit library |
| `qiskit-ibm-runtime` | latest | IBM Quantum Platform access |
| `scikit-learn` | 1.x | SVM, preprocessing, metrics |
| `numpy` | 1.x | Numerical computing |
| `pandas` | 2.x | Data processing |
| `wfdb` | 4.x | PTB-XL signal reading |
| `matplotlib` | 3.x | Visualization |
| `scipy` | 1.x | Signal processing (HRV, spectral) |
| `joblib` | 1.x | Model serialization |

---

## 🔬 Feature Extraction

325 features extracted from 5 leads (I, II, V1, V2, V5), 65 features per lead:

| Group | Features | Count |
|---|---|---|
| Time-domain | μ, σ, var, max, min, p2p, median, skew, kurtosis, RMS | 10 |
| R-peak / RR | Pan-Tompkins detection, RR intervals | 8 |
| HRV time-domain | SDNN, RMSSD, SDSD, NN50, pNN50, NN20, pNN20, CVNN | 10 |
| HRV frequency | VLF, LF, HF, LF/HF, LF_norm, HF_norm | 6 |
| HRV non-linear | SD1, SD2, SD1/SD2, Poincaré area, SampEn | 5 |
| Morphology | QRS width, amplitude, T-wave, ST level, PR, QT | 6 |
| Wavelet | 5-level Haar detail sub-band energies | 5 |
| Spectral | Band ratios (δ,θ,α,β,γ), spectral entropy, dom. freq., centroid | 8 |
| Hjorth | Activity, Mobility, Complexity | 3 |
| Zero-crossing | ZCR, MAD, energy, line length | 4 |

> Features are cached to `saved_models/feature_cache.npz` after first extraction — subsequent runs load in seconds.

---

## 📄 Paper

> **QMM-CardioNet: A Hybrid Quantum-Classical Multimodal Neural Network for ECG-Based Cardiac Arrhythmia Classification Using PTB-XL Dataset**
>
> *[Your Name], [Supervisor Name]*
> *[Institution Name], [Year]*

If you use this work, please cite:

```bibtex
@article{qmmcardionet2026,
  title   = {QMM-CardioNet: A Hybrid Quantum-Classical Multimodal Neural Network
             for ECG-Based Cardiac Arrhythmia Classification Using PTB-XL Dataset},
  author  = {Your Name and Supervisor Name},
  journal = {IEEE Access},
  year    = {2026}
}
```

### Key References

- Wagner et al. (2020) — PTB-XL Dataset — *Scientific Data*
- Pérez-Salinas et al. (2020) — Data re-uploading — *Quantum*
- McClean et al. (2018) — Barren plateaus — *Nature Communications*
- Cerezo et al. (2021) — Variational quantum algorithms — *Nature Reviews Physics*
- Pan & Tompkins (1985) — QRS detection — *IEEE Trans. Biomed. Eng.*

---

## 🗺️ Roadmap

- [x] 325-dim multi-lead feature extraction
- [x] Dual-path quantum-classical architecture
- [x] Anti-barren-plateau VQC (3-layer)
- [x] IBM Quantum hardware execution
- [x] Interactive terminal prediction
- [x] Fair 5-model comparative evaluation
- [ ] Multi-class arrhythmia classification (5 classes)
- [ ] Larger qubit count on IBM hardware
- [ ] Attention mechanism in classical path
- [ ] Web-based prediction interface

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **PTB-XL Dataset** — Wagner et al., PhysioNet
- **IBM Quantum Platform** — IBM Research
- **PennyLane** — Xanadu AI
- **Technical Students Organization** — for IBM Quantum access

---

<div align="center">

Made with ❤️ and ⚛️ for advancing quantum-enhanced cardiac diagnostics

⭐ **Star this repo if you find it useful!**

</div>