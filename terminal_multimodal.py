
import os
import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
MODEL_PATH  = "saved_models/best_multimodal_model.pth"
CACHE_PATH  = "saved_models/feature_cache.npz"   # used to get scaler
device      = "cpu"

# ── colour codes for terminal ──
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════
#  BANNER
# ═══════════════════════════════════════════════════════════
def print_banner():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║         QMM CARDIONET2 — ECG Prediction Terminal         ║
║         Quantum Multimodal Neural Network                ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")


# ═══════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════
def load_model():
    """Load the trained quantum multimodal model."""
    try:
        from models.multimodal_model import DualPathQuantumNet
    except ImportError:
        try:
            from models.multimodal_model import MultimodalQuantumNet as DualPathQuantumNet
        except ImportError:
            print(f"{RED}ERROR: Cannot import model class.{RESET}")
            print("Make sure train/train_multimodal.py exists.")
            sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"{RED}ERROR: Model not found at {MODEL_PATH}{RESET}")
        print("Train the model first:  python train/train_multimodal.py")
        sys.exit(1)

    # get ecg_dim from cache
    if os.path.exists(CACHE_PATH):
        data    = np.load(CACHE_PATH)
        ecg_dim = data["ecg"].shape[1]
    else:
        ecg_dim = 325   # default: 5 leads × 65 features

    model = DualPathQuantumNet(ecg_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print(f"{GREEN}✓ Model loaded from {MODEL_PATH}{RESET}")
    print(f"  ECG feature dim : {ecg_dim}")
    return model, ecg_dim


# ═══════════════════════════════════════════════════════════
#  GET SCALER FROM CACHE
# ═══════════════════════════════════════════════════════════
def get_scaler_from_cache():
    import joblib

    # use shared scaler — same one used during training
    if os.path.exists("saved_models/shared_scaler.pkl"):
        scaler  = joblib.load("saved_models/shared_scaler.pkl")
        ecg_dim = scaler.n_features_in_ - 2
        print(f"{GREEN}✓ Loaded shared scaler{RESET}")
        return scaler, ecg_dim

    # fallback if shared scaler not found
    if os.path.exists(CACHE_PATH):
        from sklearn.preprocessing import StandardScaler
        data     = np.load(CACHE_PATH)
        ecg_arr  = data["ecg"]
        clin_arr = data["clin"]
        combined = np.concatenate([ecg_arr, clin_arr], axis=1)
        scaler   = StandardScaler()
        scaler.fit(combined)
        ecg_dim  = ecg_arr.shape[1]
        print(f"{YELLOW}⚠ Using rebuilt scaler — run train_all_flat.py for best results{RESET}")
        return scaler, ecg_dim

    print(f"{YELLOW}⚠ No scaler found{RESET}")
    return None, 325


# ═══════════════════════════════════════════════════════════
#  GET USER INPUT
# ═══════════════════════════════════════════════════════════
def get_patient_input():
    """Interactively collect patient data from terminal."""

    print(f"\n{BOLD}{'─'*58}{RESET}")
    print(f"{BOLD}  Enter Patient Data{RESET}")
    print(f"{BOLD}{'─'*58}{RESET}\n")

    # ── Age ──
    while True:
        try:
            age = float(input(f"  {CYAN}Age{RESET} (years, e.g. 55): ").strip())
            if 0 < age < 120:
                break
            print(f"  {YELLOW}Please enter a valid age (1-120){RESET}")
        except ValueError:
            print(f"  {YELLOW}Please enter a number{RESET}")

    # ── Sex ──
    while True:
        sex_input = input(f"  {CYAN}Sex{RESET} (0=Female, 1=Male): ").strip()
        if sex_input in ["0", "1"]:
            sex = float(sex_input)
            break
        # also accept text input
        if sex_input.lower() in ["f", "female"]:
            sex = 0.0; break
        if sex_input.lower() in ["m", "male"]:
            sex = 1.0; break
        print(f"  {YELLOW}Please enter 0 (Female) or 1 (Male){RESET}")

    # ── ECG Path ──
    print(f"\n  {CYAN}ECG File Path{RESET}")
    print(f"  Enter path WITHOUT extension")
    print(f"  Example: data/ptbxl/records500/00000/00001_hr")

    while True:
        ecg_path = input(f"\n  Path: ").strip()

        # remove extension if accidentally included
        ecg_path = ecg_path.replace(".hea", "").replace(".dat", "")

        # check if file exists
        if os.path.exists(ecg_path + ".hea"):
            break
        # try with .hea check relaxed
        if os.path.exists(ecg_path):
            break
        print(f"  {YELLOW}File not found: {ecg_path}.hea{RESET}")
        retry = input(f"  Try again? (y/n): ").strip().lower()
        if retry != "y":
            print(f"  {YELLOW}Skipping ECG file — using zero signal{RESET}")
            ecg_path = None
            break

    # ── IBM Circuit ──
    print(f"\n  {CYAN}Draw IBM Quantum Circuit?{RESET}")
    ibm_input = input("  Show IBM circuit diagram? (y/n): ").strip().lower()
    draw_ibm  = ibm_input == "y"

    return {
        "age"      : age,
        "sex"      : sex,
        "ecg_path" : ecg_path,
        "draw_ibm" : draw_ibm,
    }


# ═══════════════════════════════════════════════════════════
#  EXTRACT FEATURES
# ═══════════════════════════════════════════════════════════
def extract_features(patient):
    """Extract ECG features from patient data."""
    from dataset.feature_extractor import (
        extract_ecg_features_multilead,
        extract_ecg_features
    )

    print(f"\n{CYAN}Extracting ECG features …{RESET}")

    if patient["ecg_path"] is not None:
        try:
            import wfdb
            signal, _ = wfdb.rdsamp(patient["ecg_path"])
            if signal.shape[1] >= 11:
                ecg_feats = extract_ecg_features_multilead(signal)
            else:
                ecg_feats = extract_ecg_features(signal[:, 0])
            print(f"  {GREEN}✓ ECG loaded: {signal.shape} → {len(ecg_feats)} features{RESET}")
        except Exception as e:
            print(f"  {YELLOW}⚠ ECG load failed ({e}) — using zero features{RESET}")
            ecg_feats = np.zeros(325, dtype=np.float32)
    else:
        ecg_feats = np.zeros(325, dtype=np.float32)

    clin_feats = np.array([patient["age"], patient["sex"]], dtype=np.float32)

    return ecg_feats, clin_feats


# ═══════════════════════════════════════════════════════════
#  PREDICT
# ═══════════════════════════════════════════════════════════
def predict(model, ecg_feats, clin_feats, scaler, ecg_dim):
    """Run model prediction and return confidence score."""

    # combine and scale
    combined = np.concatenate([ecg_feats, clin_feats]).reshape(1, -1)

    if scaler is not None:
        # pad/trim to match scaler's expected shape
        expected = scaler.n_features_in_
        actual   = combined.shape[1]
        if actual < expected:
            combined = np.pad(combined, ((0,0),(0, expected-actual)))
        elif actual > expected:
            combined = combined[:, :expected]
        combined = scaler.transform(combined).astype(np.float32)

    combined = np.nan_to_num(combined, nan=0., posinf=0., neginf=0.)

    ecg_t  = torch.tensor(combined[:, :ecg_dim],  dtype=torch.float32)
    clin_t = torch.tensor(combined[:, ecg_dim:ecg_dim+2], dtype=torch.float32)

    with torch.no_grad():
        logit      = model(ecg_t, clin_t)
        confidence = torch.sigmoid(logit).item()

    return confidence


# ═══════════════════════════════════════════════════════════
#  DISPLAY RESULT
# ═══════════════════════════════════════════════════════════
def display_result(patient, confidence):
    """Display prediction result with confidence bar."""

    prediction = "ABNORMAL" if confidence >= 0.5 else "NORMAL"
    color      = RED if prediction == "ABNORMAL" else GREEN

    # confidence bar
    bar_len  = 40
    filled   = int(confidence * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  PREDICTION RESULT{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}\n")

    print(f"  Patient   :  Age={patient['age']:.0f}  "
          f"Sex={'Male' if patient['sex']==1 else 'Female'}")
    print(f"  ECG File  :  {patient['ecg_path'] or 'Not provided'}")

    print(f"\n  Diagnosis :  {color}{BOLD}{prediction}{RESET}")
    print(f"  Confidence:  {color}{BOLD}{confidence:.4f}  ({confidence*100:.1f}%){RESET}")
    print(f"\n  [{bar}]  {confidence*100:.1f}%")

    # risk interpretation
    print(f"\n  {'─'*54}")
    if confidence >= 0.85:
        print(f"  {RED}⚠  HIGH RISK — Strong evidence of cardiac abnormality{RESET}")
        print(f"  {RED}   Immediate clinical evaluation recommended{RESET}")
    elif confidence >= 0.65:
        print(f"  {YELLOW}⚠  MODERATE RISK — Possible cardiac abnormality{RESET}")
        print(f"  {YELLOW}   Clinical review recommended{RESET}")
    elif confidence >= 0.5:
        print(f"  {YELLOW}ℹ  LOW-MODERATE RISK — Borderline result{RESET}")
        print(f"  {YELLOW}   Follow-up advised{RESET}")
    elif confidence >= 0.3:
        print(f"  {GREEN}ℹ  LOW RISK — Likely normal ECG{RESET}")
        print(f"  {GREEN}   Routine monitoring recommended{RESET}")
    else:
        print(f"  {GREEN}✓  VERY LOW RISK — Strong evidence of normal ECG{RESET}")
        print(f"  {GREEN}   No immediate action required{RESET}")

    print(f"  {'─'*54}")
    print(f"\n  {YELLOW}⚕ DISCLAIMER: This is a research tool only.{RESET}")
    print(f"  {YELLOW}  Not for clinical diagnosis. Consult a cardiologist.{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}\n")


# ═══════════════════════════════════════════════════════════
#  IBM CIRCUIT VISUALIZATION
# ═══════════════════════════════════════════════════════════
def draw_ibm_circuit_diagram(ecg_feats):
    """Draw quantum circuit using IBM Qiskit."""

    print(f"\n{CYAN}Drawing IBM Quantum Circuit …{RESET}")

    try:
        from qiskit import QuantumCircuit
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{YELLOW}Qiskit not installed.{RESET}")
        print(f"Run:  pip install qiskit pylatexenc")
        _draw_pennylane_circuit(ecg_feats)
        return

    N_QUBITS = 8
    N_LAYERS = 5

    # scale input features to [-π, π] for angle encoding
    sample   = np.tanh(ecg_feats[:N_QUBITS]) * np.pi

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    # ── Initial Hadamard ──
    for i in range(N_QUBITS):
        qc.h(i)
    qc.barrier()

    # ── Data Re-uploading Layers ──
    for layer in range(N_LAYERS):

        # AngleEmbedding — RY gates with actual input values
        for i in range(N_QUBITS):
            angle = float(sample[i]) if i < len(sample) else 0.0
            qc.ry(round(angle, 3), i)

        qc.barrier()

        # Variational layer — RZ RY RZ rotations
        for i in range(N_QUBITS):
            qc.rz(np.pi/4, i)
            qc.ry(np.pi/4, i)
            qc.rz(np.pi/4, i)

        # Entangling CNOTs (ring)
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)

        # Long-range CNOTs
        for i in range(N_QUBITS):
            qc.cx(i, (i + 2) % N_QUBITS)

        qc.barrier()

    # ── Measure ──
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    # ── Print circuit stats ──
    print(f"\n  {GREEN}IBM Quantum Circuit Stats:{RESET}")
    print(f"  Qubits        : {N_QUBITS}")
    print(f"  Layers        : {N_LAYERS}")
    print(f"  Circuit depth : {qc.depth()}")
    print(f"  Total gates   : {sum(qc.count_ops().values())}")
    print(f"  Gate types    : {dict(qc.count_ops())}")

    # ── Draw and save ──
    save_path = "ibm_quantum_circuit.png"

    try:
        fig = qc.draw(
            output  = "mpl",
            style   = "clifford",   # IBM blue theme
            fold    = 40,
            scale   = 0.65,
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  {GREEN}✓ Circuit saved → {save_path}{RESET}")
        plt.show()
    except Exception as e:
        print(f"  {YELLOW}mpl draw failed ({e}), trying text draw …{RESET}")
        print(qc.draw(output="text", fold=80))

    return qc


def _draw_pennylane_circuit(ecg_feats):
    """Fallback: print PennyLane circuit in terminal."""
    try:
        import pennylane as qml
        from models.quantum_layer import circuit, N_QUBITS, N_LAYERS
        sample  = np.tanh(ecg_feats[:N_QUBITS]).astype(np.float32)
        weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
        print("\nPennyLane Circuit:")
        print("=" * 50)
        print(qml.draw(circuit)(
            torch.tensor(sample),
            torch.tensor(weights)
        ))
        print("=" * 50)
    except Exception as e:
        print(f"{YELLOW}Circuit draw failed: {e}{RESET}")


# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════
def main():
    print_banner()

    # load model + scaler
    model, ecg_dim   = load_model()
    scaler, _        = get_scaler_from_cache()

    while True:
        try:
            # get patient input
            patient = get_patient_input()

            # extract features
            ecg_feats, clin_feats = extract_features(patient)

            # predict
            print(f"\n{CYAN}Running Quantum Multimodal inference …{RESET}")
            t0         = time.time()
            confidence = predict(model, ecg_feats, clin_feats, scaler, ecg_dim)
            inf_time   = time.time() - t0
            print(f"  {GREEN}✓ Inference done in {inf_time:.2f}s{RESET}")

            # display result
            display_result(patient, confidence)

            # IBM circuit
            if patient["draw_ibm"]:
                draw_ibm_circuit_diagram(ecg_feats)

            # ask to run again
            print(f"\n{'─'*58}")
            again = input(f"  Test another patient? (y/n): ").strip().lower()
            if again != "y":
                print(f"\n{GREEN}Thank you for using QMM CARDIONET2!{RESET}\n")
                break

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Interrupted by user.{RESET}\n")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            import traceback; traceback.print_exc()
            again = input("Try again? (y/n): ").strip().lower()
            if again != "y":
                break


if __name__ == "__main__":
    main()








# """
# terminal_multimodal.py  —  Interactive ECG Prediction Terminal
# ==============================================================
# Features:
#   1. Input patient data (age, sex, ECG path) via terminal
#   2. Extract 325 features from ECG
#   3. Run through trained Quantum Multimodal model
#   4. Show prediction + confidence score
#   5. Draw IBM Quantum circuit diagram (optional)

# Run:  python terminal_multimodal.py
# """

# import os
# import sys
# import time
# import numpy as np
# import torch
# import warnings
# warnings.filterwarnings("ignore")

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# MODEL_PATH  = "saved_models/best_multimodal_model.pth"
# CACHE_PATH  = "saved_models/feature_cache.npz"   # used to get scaler
# device      = "cpu"

# # ── colour codes for terminal ──
# GREEN  = "\033[92m"
# RED    = "\033[91m"
# YELLOW = "\033[93m"
# CYAN   = "\033[96m"
# BOLD   = "\033[1m"
# RESET  = "\033[0m"


# # ═══════════════════════════════════════════════════════════
# #  BANNER
# # ═══════════════════════════════════════════════════════════
# def print_banner():
#     print(f"""
# {CYAN}{BOLD}
# ╔══════════════════════════════════════════════════════════╗
# ║         QMM CARDIONET2 — ECG Prediction Terminal         ║
# ║         Quantum Multimodal Neural Network                ║
# ╚══════════════════════════════════════════════════════════╝
# {RESET}""")


# # ═══════════════════════════════════════════════════════════
# #  LOAD MODEL
# # ═══════════════════════════════════════════════════════════
# def load_model():
#     """Load the trained quantum multimodal model."""
#     try:
#         from models.multimodal_model import DualPathQuantumNet
#     except ImportError:
#         try:
#             from models.multimodal_model import MultimodalQuantumNet as DualPathQuantumNet
#         except ImportError:
#             print(f"{RED}ERROR: Cannot import model class.{RESET}")
#             print("Make sure train/train_multimodal.py exists.")
#             sys.exit(1)

#     if not os.path.exists(MODEL_PATH):
#         print(f"{RED}ERROR: Model not found at {MODEL_PATH}{RESET}")
#         print("Train the model first:  python train/train_multimodal.py")
#         sys.exit(1)

#     # get ecg_dim from cache
#     if os.path.exists(CACHE_PATH):
#         data    = np.load(CACHE_PATH)
#         ecg_dim = data["ecg"].shape[1]
#     else:
#         ecg_dim = 325   # default: 5 leads × 65 features

#     model = DualPathQuantumNet(ecg_dim)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()

#     print(f"{GREEN}✓ Model loaded from {MODEL_PATH}{RESET}")
#     print(f"  ECG feature dim : {ecg_dim}")
#     return model, ecg_dim


# # ═══════════════════════════════════════════════════════════
# #  GET SCALER FROM CACHE
# # ═══════════════════════════════════════════════════════════
# def get_scaler_from_cache():
#     import joblib

#     # use shared scaler — same one used during training
#     if os.path.exists("saved_models/shared_scaler.pkl"):
#         scaler  = joblib.load("saved_models/shared_scaler.pkl")
#         ecg_dim = scaler.n_features_in_ - 2
#         print(f"{GREEN}✓ Loaded shared scaler{RESET}")
#         return scaler, ecg_dim

#     # fallback if shared scaler not found
#     if os.path.exists(CACHE_PATH):
#         from sklearn.preprocessing import StandardScaler
#         data     = np.load(CACHE_PATH)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         scaler.fit(combined)
#         ecg_dim  = ecg_arr.shape[1]
#         print(f"{YELLOW}⚠ Using rebuilt scaler — run train_all_flat.py for best results{RESET}")
#         return scaler, ecg_dim

#     print(f"{YELLOW}⚠ No scaler found{RESET}")
#     return None, 325


# # ═══════════════════════════════════════════════════════════
# #  GET USER INPUT
# # ═══════════════════════════════════════════════════════════
# def get_patient_input():
#     """Interactively collect patient data from terminal."""

#     print(f"\n{BOLD}{'─'*58}{RESET}")
#     print(f"{BOLD}  Enter Patient Data{RESET}")
#     print(f"{BOLD}{'─'*58}{RESET}\n")

#     # ── Age ──
#     while True:
#         try:
#             age = float(input(f"  {CYAN}Age{RESET} (years, e.g. 55): ").strip())
#             if 0 < age < 120:
#                 break
#             print(f"  {YELLOW}Please enter a valid age (1-120){RESET}")
#         except ValueError:
#             print(f"  {YELLOW}Please enter a number{RESET}")

#     # ── Sex ──
#     while True:
#         sex_input = input(f"  {CYAN}Sex{RESET} (0=Female, 1=Male): ").strip()
#         if sex_input in ["0", "1"]:
#             sex = float(sex_input)
#             break
#         # also accept text input
#         if sex_input.lower() in ["f", "female"]:
#             sex = 0.0; break
#         if sex_input.lower() in ["m", "male"]:
#             sex = 1.0; break
#         print(f"  {YELLOW}Please enter 0 (Female) or 1 (Male){RESET}")

#     # ── ECG Path ──
#     print(f"\n  {CYAN}ECG File Path{RESET}")
#     print(f"  Enter path WITHOUT extension")
#     print(f"  Example: data/ptbxl/records500/00000/00001_hr")

#     while True:
#         ecg_path = input(f"\n  Path: ").strip()

#         # remove extension if accidentally included
#         ecg_path = ecg_path.replace(".hea", "").replace(".dat", "")

#         # check if file exists
#         if os.path.exists(ecg_path + ".hea"):
#             break
#         # try with .hea check relaxed
#         if os.path.exists(ecg_path):
#             break
#         print(f"  {YELLOW}File not found: {ecg_path}.hea{RESET}")
#         retry = input(f"  Try again? (y/n): ").strip().lower()
#         if retry != "y":
#             print(f"  {YELLOW}Skipping ECG file — using zero signal{RESET}")
#             ecg_path = None
#             break

#     # ── IBM Circuit ──
#     print(f"\n  {CYAN}Draw IBM Quantum Circuit?{RESET}")
#     ibm_input = input("  Show IBM circuit diagram? (y/n): ").strip().lower()
#     draw_ibm  = ibm_input == "y"

#     return {
#         "age"      : age,
#         "sex"      : sex,
#         "ecg_path" : ecg_path,
#         "draw_ibm" : draw_ibm,
#     }


# # ═══════════════════════════════════════════════════════════
# #  EXTRACT FEATURES
# # ═══════════════════════════════════════════════════════════
# def extract_features(patient):
#     """Extract ECG features from patient data."""
#     from dataset.feature_extractor import (
#         extract_ecg_features_multilead,
#         extract_ecg_features
#     )

#     print(f"\n{CYAN}Extracting ECG features …{RESET}")

#     if patient["ecg_path"] is not None:
#         try:
#             import wfdb
#             signal, _ = wfdb.rdsamp(patient["ecg_path"])
#             if signal.shape[1] >= 11:
#                 ecg_feats = extract_ecg_features_multilead(signal)
#             else:
#                 ecg_feats = extract_ecg_features(signal[:, 0])
#             print(f"  {GREEN}✓ ECG loaded: {signal.shape} → {len(ecg_feats)} features{RESET}")
#         except Exception as e:
#             print(f"  {YELLOW}⚠ ECG load failed ({e}) — using zero features{RESET}")
#             ecg_feats = np.zeros(325, dtype=np.float32)
#     else:
#         ecg_feats = np.zeros(325, dtype=np.float32)

#     clin_feats = np.array([patient["age"], patient["sex"]], dtype=np.float32)

#     return ecg_feats, clin_feats


# # ═══════════════════════════════════════════════════════════
# #  PREDICT
# # ═══════════════════════════════════════════════════════════
# def predict(model, ecg_feats, clin_feats, scaler, ecg_dim):
#     """Run model prediction and return confidence score."""

#     # combine and scale
#     combined = np.concatenate([ecg_feats, clin_feats]).reshape(1, -1)

#     if scaler is not None:
#         # pad/trim to match scaler's expected shape
#         expected = scaler.n_features_in_
#         actual   = combined.shape[1]
#         if actual < expected:
#             combined = np.pad(combined, ((0,0),(0, expected-actual)))
#         elif actual > expected:
#             combined = combined[:, :expected]
#         combined = scaler.transform(combined).astype(np.float32)

#     combined = np.nan_to_num(combined, nan=0., posinf=0., neginf=0.)

#     ecg_t  = torch.tensor(combined[:, :ecg_dim],  dtype=torch.float32)
#     clin_t = torch.tensor(combined[:, ecg_dim:ecg_dim+2], dtype=torch.float32)

#     with torch.no_grad():
#         logit      = model(ecg_t, clin_t)
#         confidence = torch.sigmoid(logit).item()

#     return confidence


# # ═══════════════════════════════════════════════════════════
# #  DISPLAY RESULT
# # ═══════════════════════════════════════════════════════════
# def display_result(patient, confidence):
#     """Display prediction result with confidence bar."""

#     prediction = "ABNORMAL" if confidence >= 0.5 else "NORMAL"
#     color      = RED if prediction == "ABNORMAL" else GREEN

#     # confidence bar
#     bar_len  = 40
#     filled   = int(confidence * bar_len)
#     bar      = "█" * filled + "░" * (bar_len - filled)

#     print(f"\n{BOLD}{'═'*58}{RESET}")
#     print(f"{BOLD}  PREDICTION RESULT{RESET}")
#     print(f"{BOLD}{'═'*58}{RESET}\n")

#     print(f"  Patient   :  Age={patient['age']:.0f}  "
#           f"Sex={'Male' if patient['sex']==1 else 'Female'}")
#     print(f"  ECG File  :  {patient['ecg_path'] or 'Not provided'}")

#     print(f"\n  Diagnosis :  {color}{BOLD}{prediction}{RESET}")
#     print(f"  Confidence:  {color}{BOLD}{confidence:.4f}  ({confidence*100:.1f}%){RESET}")
#     print(f"\n  [{bar}]  {confidence*100:.1f}%")

#     # risk interpretation
#     print(f"\n  {'─'*54}")
#     if confidence >= 0.85:
#         print(f"  {RED}⚠  HIGH RISK — Strong evidence of cardiac abnormality{RESET}")
#         print(f"  {RED}   Immediate clinical evaluation recommended{RESET}")
#     elif confidence >= 0.65:
#         print(f"  {YELLOW}⚠  MODERATE RISK — Possible cardiac abnormality{RESET}")
#         print(f"  {YELLOW}   Clinical review recommended{RESET}")
#     elif confidence >= 0.5:
#         print(f"  {YELLOW}ℹ  LOW-MODERATE RISK — Borderline result{RESET}")
#         print(f"  {YELLOW}   Follow-up advised{RESET}")
#     elif confidence >= 0.3:
#         print(f"  {GREEN}ℹ  LOW RISK — Likely normal ECG{RESET}")
#         print(f"  {GREEN}   Routine monitoring recommended{RESET}")
#     else:
#         print(f"  {GREEN}✓  VERY LOW RISK — Strong evidence of normal ECG{RESET}")
#         print(f"  {GREEN}   No immediate action required{RESET}")

#     print(f"  {'─'*54}")
#     print(f"\n  {YELLOW}⚕ DISCLAIMER: This is a research tool only.{RESET}")
#     print(f"  {YELLOW}  Not for clinical diagnosis. Consult a cardiologist.{RESET}")
#     print(f"{BOLD}{'═'*58}{RESET}\n")


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit_diagram(ecg_feats):
#     """
#     Draw quantum circuit using IBM Qiskit.
#     Also submits circuit to IBM Quantum backend if API key is configured.

#     To use IBM Quantum backend:
#       1. Run once to save account:
#          from qiskit_ibm_runtime import QiskitRuntimeService
#          QiskitRuntimeService.save_account(
#              channel="ibm_quantum",
#              token="YOUR_API_KEY_HERE",
#              overwrite=True
#          )
#       2. Then run terminal_multimodal.py normally.
#          It will detect saved credentials automatically.
#     """

#     print(f"\n{CYAN}Drawing IBM Quantum Circuit …{RESET}")

#     try:
#         from qiskit import QuantumCircuit, transpile
#         import matplotlib.pyplot as plt
#     except ImportError:
#         print(f"{YELLOW}Qiskit not installed.{RESET}")
#         print(f"Run:  pip install qiskit qiskit-ibm-runtime pylatexenc")
#         _draw_pennylane_circuit(ecg_feats)
#         return

#     N_QUBITS = 8
#     N_LAYERS = 3   # anti-barren-plateau: 3 layers

#     # scale input features for angle encoding
#     sample = np.tanh(ecg_feats[:N_QUBITS]) * np.pi

#     # ── Build circuit ──────────────────────────────────────
#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)
#     qc.h(range(N_QUBITS))
#     qc.barrier()

#     for layer in range(N_LAYERS):
#         # Data re-uploading: RY encoding at every layer
#         for i in range(N_QUBITS):
#             angle = float(sample[i]) if i < len(sample) else 0.0
#             qc.ry(round(angle, 3), i)
#         qc.barrier()

#         # Variational rotations: RZ-RY-RZ per qubit
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)

#         # CNOT ring entanglement
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)
#         qc.barrier()

#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     # ── Print stats ────────────────────────────────────────
#     print(f"\n  {GREEN}IBM Quantum Circuit Stats:{RESET}")
#     print(f"  Qubits        : {N_QUBITS}")
#     print(f"  Layers        : {N_LAYERS}  (anti-barren-plateau design)")
#     print(f"  Circuit depth : {qc.depth()}")
#     print(f"  Total gates   : {sum(qc.count_ops().values())}")
#     print(f"  Gate types    : {dict(qc.count_ops())}")

#     # ── Draw and save locally ──────────────────────────────
#     save_path = "ibm_quantum_circuit.png"
#     try:
#         fig = qc.draw(
#             output = "mpl",
#             style  = "clifford",
#             fold   = 40,
#             scale  = 0.7,
#         )
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"\n  {GREEN}✓ Circuit diagram saved → {save_path}{RESET}")
#         plt.close()
#     except Exception as e:
#         print(f"  {YELLOW}mpl draw failed ({e}){RESET}")
#         print(qc.draw(output="text", fold=80))

#     # ── IBM Quantum Backend (optional) ────────────────────
#     print(f"\n  {CYAN}Submit to IBM Quantum backend? (y/n): {RESET}", end="")
#     ibm_run = input().strip().lower()

#     if ibm_run == "y":
#         _run_on_ibm_backend(qc, N_QUBITS)

#     return qc


# def _run_on_ibm_backend(qc, N_QUBITS):
#     """
#     Submit circuit to IBM Quantum backend using saved credentials.
#     Saves results to ibm_quantum_results.txt
#     """
#     try:
#         from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
#         from qiskit import transpile
#     except ImportError:
#         print(f"{YELLOW}qiskit-ibm-runtime not installed.")
#         print(f"Run:  pip install qiskit-ibm-runtime{RESET}")
#         return

#     print(f"\n{CYAN}Connecting to IBM Quantum …{RESET}")

#     # ── Check for saved account ────────────────────────────
#     try:
#         service = QiskitRuntimeService()
#         print(f"  {GREEN}✓ IBM account loaded from saved credentials{RESET}")
#     except Exception:
#         print(f"  {YELLOW}No saved IBM account found.{RESET}")
#         print(f"  To save your account, run once:")
#         print(f"  {CYAN}from qiskit_ibm_runtime import QiskitRuntimeService")
#         print(f"  QiskitRuntimeService.save_account(")
#         print(f"      channel='ibm_quantum',")
#         print(f"      token='YOUR_API_KEY_HERE',")
#         print(f"      overwrite=True)")
#         print(f"  {RESET}")
#         return

#     # ── Select backend ─────────────────────────────────────
#     try:
#         backends = service.backends(
#             filters=lambda b: b.configuration().n_qubits >= N_QUBITS
#                               and not b.configuration().simulator
#                               and b.status().operational
#         )
#         if not backends:
#             print(f"  {YELLOW}No real backends available. Using simulator.{RESET}")
#             backend = service.backend("ibmq_qasm_simulator")
#         else:
#             # pick least busy
#             from qiskit_ibm_provider import least_busy
#             backend = least_busy(backends)

#         print(f"  {GREEN}✓ Backend selected: {backend.name}{RESET}")
#         print(f"  Qubits : {backend.configuration().n_qubits}")
#         print(f"  Status : {backend.status().status_msg}")

#     except Exception as e:
#         print(f"  {YELLOW}Backend selection failed: {e}")
#         print(f"  Using default simulator.{RESET}")
#         try:
#             backend = service.backend("ibmq_qasm_simulator")
#         except Exception:
#             print(f"  {RED}Cannot connect to any backend.{RESET}")
#             return

#     # ── Transpile and run ──────────────────────────────────
#     print(f"\n  Transpiling circuit for {backend.name} …")
#     try:
#         tc = transpile(qc, backend=backend, optimization_level=1)
#         print(f"  {GREEN}✓ Transpiled depth: {tc.depth()}{RESET}")

#         print(f"  Submitting job (shots=1024) …")
#         sampler = Sampler(mode=backend)
#         job = sampler.run([tc], shots=1024)
#         job_id = job.job_id()
#         print(f"  {GREEN}✓ Job submitted!{RESET}")
#         print(f"  Job ID : {job_id}")
#         print(f"  Track  : https://quantum.ibm.com/jobs/{job_id}")

#         # save job info
#         with open("ibm_quantum_results.txt", "w") as f:
#             f.write(f"QMM-CardioNet IBM Quantum Job\n")
#             f.write(f"Backend : {backend.name}\n")
#             f.write(f"Job ID  : {job_id}\n")
#             f.write(f"URL     : https://quantum.ibm.com/jobs/{job_id}\n")
#             f.write(f"Qubits  : {N_QUBITS}\n")
#             f.write(f"Depth   : {tc.depth()}\n")
#         print(f"  {GREEN}✓ Job info saved → ibm_quantum_results.txt{RESET}")
#         print(f"\n  {CYAN}Note: Results appear on IBM Quantum Cloud once job completes.{RESET}")
#         print(f"  View at: https://quantum.ibm.com/jobs/{job_id}")

#     except Exception as e:
#         print(f"  {RED}Job submission failed: {e}{RESET}")
#         print(f"  Check your API key and account quota.")


# def _draw_pennylane_circuit(ecg_feats):
#     """Fallback: print PennyLane circuit in terminal."""
#     try:
#         import pennylane as qml
#         from models.quantum_layer import circuit, N_QUBITS, N_LAYERS
#         sample  = np.tanh(ecg_feats[:N_QUBITS]).astype(np.float32)
#         weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#         print("\nPennyLane Circuit:")
#         print("=" * 50)
#         print(qml.draw(circuit)(
#             torch.tensor(sample),
#             torch.tensor(weights)
#         ))
#         print("=" * 50)
#     except Exception as e:
#         print(f"{YELLOW}Circuit draw failed: {e}{RESET}")


# # ═══════════════════════════════════════════════════════════
# #  MAIN LOOP
# # ═══════════════════════════════════════════════════════════
# def main():
#     print_banner()

#     # load model + scaler
#     model, ecg_dim   = load_model()
#     scaler, _        = get_scaler_from_cache()

#     while True:
#         try:
#             # get patient input
#             patient = get_patient_input()

#             # extract features
#             ecg_feats, clin_feats = extract_features(patient)

#             # predict
#             print(f"\n{CYAN}Running Quantum Multimodal inference …{RESET}")
#             t0         = time.time()
#             confidence = predict(model, ecg_feats, clin_feats, scaler, ecg_dim)
#             inf_time   = time.time() - t0
#             print(f"  {GREEN}✓ Inference done in {inf_time:.2f}s{RESET}")

#             # display result
#             display_result(patient, confidence)

#             # IBM circuit
#             if patient["draw_ibm"]:
#                 draw_ibm_circuit_diagram(ecg_feats)

#             # ask to run again
#             print(f"\n{'─'*58}")
#             again = input(f"  Test another patient? (y/n): ").strip().lower()
#             if again != "y":
#                 print(f"\n{GREEN}Thank you for using QMM CARDIONET2!{RESET}\n")
#                 break

#         except KeyboardInterrupt:
#             print(f"\n\n{YELLOW}Interrupted by user.{RESET}\n")
#             break
#         except Exception as e:
#             print(f"\n{RED}Error: {e}{RESET}")
#             import traceback; traceback.print_exc()
#             again = input("Try again? (y/n): ").strip().lower()
#             if again != "y":
#                 break


# if __name__ == "__main__":
#     main()




# """
# terminal_multimodal.py  —  Interactive ECG Prediction Terminal
# ==============================================================
# Features:
#   1. Input patient data (age, sex, ECG path) via terminal
#   2. Extract 325 features from ECG
#   3. Run through trained Quantum Multimodal model
#   4. Show prediction + confidence score
#   5. Draw IBM Quantum circuit diagram (optional)

# Run:  python terminal_multimodal.py
# """

# import os
# import sys
# import time
# import numpy as np
# import torch
# import warnings
# warnings.filterwarnings("ignore")

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# MODEL_PATH  = "saved_models/best_multimodal_model.pth"
# CACHE_PATH  = "saved_models/feature_cache.npz"   # used to get scaler
# device      = "cpu"

# # ── colour codes for terminal ──
# GREEN  = "\033[92m"
# RED    = "\033[91m"
# YELLOW = "\033[93m"
# CYAN   = "\033[96m"
# BOLD   = "\033[1m"
# RESET  = "\033[0m"


# # ═══════════════════════════════════════════════════════════
# #  BANNER
# # ═══════════════════════════════════════════════════════════
# def print_banner():
#     print(f"""
# {CYAN}{BOLD}
# ╔══════════════════════════════════════════════════════════╗
# ║         QMM CARDIONET2 — ECG Prediction Terminal         ║
# ║         Quantum Multimodal Neural Network                ║
# ╚══════════════════════════════════════════════════════════╝
# {RESET}""")


# # ═══════════════════════════════════════════════════════════
# #  LOAD MODEL
# # ═══════════════════════════════════════════════════════════
# def load_model():
#     """Load the trained quantum multimodal model."""
#     try:
#         from models.multimodal_model import DualPathQuantumNet
#     except ImportError:
#         try:
#             from models.multimodal_model import MultimodalQuantumNet as DualPathQuantumNet
#         except ImportError:
#             print(f"{RED}ERROR: Cannot import model class.{RESET}")
#             print("Make sure train/train_multimodal.py exists.")
#             sys.exit(1)

#     if not os.path.exists(MODEL_PATH):
#         print(f"{RED}ERROR: Model not found at {MODEL_PATH}{RESET}")
#         print("Train the model first:  python train/train_multimodal.py")
#         sys.exit(1)

#     # get ecg_dim from cache
#     if os.path.exists(CACHE_PATH):
#         data    = np.load(CACHE_PATH)
#         ecg_dim = data["ecg"].shape[1]
#     else:
#         ecg_dim = 325   # default: 5 leads × 65 features

#     model = DualPathQuantumNet(ecg_dim)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()

#     print(f"{GREEN}✓ Model loaded from {MODEL_PATH}{RESET}")
#     print(f"  ECG feature dim : {ecg_dim}")
#     return model, ecg_dim


# # ═══════════════════════════════════════════════════════════
# #  GET SCALER FROM CACHE
# # ═══════════════════════════════════════════════════════════
# def get_scaler_from_cache():
#     import joblib

#     # use shared scaler — same one used during training
#     if os.path.exists("saved_models/shared_scaler.pkl"):
#         scaler  = joblib.load("saved_models/shared_scaler.pkl")
#         ecg_dim = scaler.n_features_in_ - 2
#         print(f"{GREEN}✓ Loaded shared scaler{RESET}")
#         return scaler, ecg_dim

#     # fallback if shared scaler not found
#     if os.path.exists(CACHE_PATH):
#         from sklearn.preprocessing import StandardScaler
#         data     = np.load(CACHE_PATH)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         combined = np.concatenate([ecg_arr, clin_arr], axis=1)
#         scaler   = StandardScaler()
#         scaler.fit(combined)
#         ecg_dim  = ecg_arr.shape[1]
#         print(f"{YELLOW}⚠ Using rebuilt scaler — run train_all_flat.py for best results{RESET}")
#         return scaler, ecg_dim

#     print(f"{YELLOW}⚠ No scaler found{RESET}")
#     return None, 325


# # ═══════════════════════════════════════════════════════════
# #  GET USER INPUT
# # ═══════════════════════════════════════════════════════════
# def get_patient_input():
#     """Interactively collect patient data from terminal."""

#     print(f"\n{BOLD}{'─'*58}{RESET}")
#     print(f"{BOLD}  Enter Patient Data{RESET}")
#     print(f"{BOLD}{'─'*58}{RESET}\n")

#     # ── Age ──
#     while True:
#         try:
#             age = float(input(f"  {CYAN}Age{RESET} (years, e.g. 55): ").strip())
#             if 0 < age < 120:
#                 break
#             print(f"  {YELLOW}Please enter a valid age (1-120){RESET}")
#         except ValueError:
#             print(f"  {YELLOW}Please enter a number{RESET}")

#     # ── Sex ──
#     while True:
#         sex_input = input(f"  {CYAN}Sex{RESET} (0=Female, 1=Male): ").strip()
#         if sex_input in ["0", "1"]:
#             sex = float(sex_input)
#             break
#         # also accept text input
#         if sex_input.lower() in ["f", "female"]:
#             sex = 0.0; break
#         if sex_input.lower() in ["m", "male"]:
#             sex = 1.0; break
#         print(f"  {YELLOW}Please enter 0 (Female) or 1 (Male){RESET}")

#     # ── ECG Path ──
#     print(f"\n  {CYAN}ECG File Path{RESET}")
#     print(f"  Enter path WITHOUT extension")
#     print(f"  Example: data/ptbxl/records500/00000/00001_hr")

#     while True:
#         ecg_path = input(f"\n  Path: ").strip()

#         # remove extension if accidentally included
#         ecg_path = ecg_path.replace(".hea", "").replace(".dat", "")

#         # check if file exists
#         if os.path.exists(ecg_path + ".hea"):
#             break
#         # try with .hea check relaxed
#         if os.path.exists(ecg_path):
#             break
#         print(f"  {YELLOW}File not found: {ecg_path}.hea{RESET}")
#         retry = input(f"  Try again? (y/n): ").strip().lower()
#         if retry != "y":
#             print(f"  {YELLOW}Skipping ECG file — using zero signal{RESET}")
#             ecg_path = None
#             break

#     # ── IBM Circuit ──
#     print(f"\n  {CYAN}Draw IBM Quantum Circuit?{RESET}")
#     ibm_input = input("  Show IBM circuit diagram? (y/n): ").strip().lower()
#     draw_ibm  = ibm_input == "y"

#     return {
#         "age"      : age,
#         "sex"      : sex,
#         "ecg_path" : ecg_path,
#         "draw_ibm" : draw_ibm,
#     }


# # ═══════════════════════════════════════════════════════════
# #  EXTRACT FEATURES
# # ═══════════════════════════════════════════════════════════
# def extract_features(patient):
#     """Extract ECG features from patient data."""
#     from dataset.feature_extractor import (
#         extract_ecg_features_multilead,
#         extract_ecg_features
#     )

#     print(f"\n{CYAN}Extracting ECG features …{RESET}")

#     if patient["ecg_path"] is not None:
#         try:
#             import wfdb
#             signal, _ = wfdb.rdsamp(patient["ecg_path"])
#             if signal.shape[1] >= 11:
#                 ecg_feats = extract_ecg_features_multilead(signal)
#             else:
#                 ecg_feats = extract_ecg_features(signal[:, 0])
#             print(f"  {GREEN}✓ ECG loaded: {signal.shape} → {len(ecg_feats)} features{RESET}")
#         except Exception as e:
#             print(f"  {YELLOW}⚠ ECG load failed ({e}) — using zero features{RESET}")
#             ecg_feats = np.zeros(325, dtype=np.float32)
#     else:
#         ecg_feats = np.zeros(325, dtype=np.float32)

#     clin_feats = np.array([patient["age"], patient["sex"]], dtype=np.float32)

#     return ecg_feats, clin_feats


# # ═══════════════════════════════════════════════════════════
# #  PREDICT
# # ═══════════════════════════════════════════════════════════
# def predict(model, ecg_feats, clin_feats, scaler, ecg_dim):
#     """Run model prediction and return confidence score."""

#     # combine and scale
#     combined = np.concatenate([ecg_feats, clin_feats]).reshape(1, -1)

#     if scaler is not None:
#         # pad/trim to match scaler's expected shape
#         expected = scaler.n_features_in_
#         actual   = combined.shape[1]
#         if actual < expected:
#             combined = np.pad(combined, ((0,0),(0, expected-actual)))
#         elif actual > expected:
#             combined = combined[:, :expected]
#         combined = scaler.transform(combined).astype(np.float32)

#     combined = np.nan_to_num(combined, nan=0., posinf=0., neginf=0.)

#     ecg_t  = torch.tensor(combined[:, :ecg_dim],  dtype=torch.float32)
#     clin_t = torch.tensor(combined[:, ecg_dim:ecg_dim+2], dtype=torch.float32)

#     with torch.no_grad():
#         logit      = model(ecg_t, clin_t)
#         confidence = torch.sigmoid(logit).item()

#     return confidence


# # ═══════════════════════════════════════════════════════════
# #  DISPLAY RESULT
# # ═══════════════════════════════════════════════════════════
# def display_result(patient, confidence):
#     """Display prediction result with confidence bar."""

#     prediction = "ABNORMAL" if confidence >= 0.5 else "NORMAL"
#     color      = RED if prediction == "ABNORMAL" else GREEN

#     # confidence bar
#     bar_len  = 40
#     filled   = int(confidence * bar_len)
#     bar      = "█" * filled + "░" * (bar_len - filled)

#     print(f"\n{BOLD}{'═'*58}{RESET}")
#     print(f"{BOLD}  PREDICTION RESULT{RESET}")
#     print(f"{BOLD}{'═'*58}{RESET}\n")

#     print(f"  Patient   :  Age={patient['age']:.0f}  "
#           f"Sex={'Male' if patient['sex']==1 else 'Female'}")
#     print(f"  ECG File  :  {patient['ecg_path'] or 'Not provided'}")

#     print(f"\n  Diagnosis :  {color}{BOLD}{prediction}{RESET}")
#     print(f"  Confidence:  {color}{BOLD}{confidence:.4f}  ({confidence*100:.1f}%){RESET}")
#     print(f"\n  [{bar}]  {confidence*100:.1f}%")

#     # risk interpretation
#     print(f"\n  {'─'*54}")
#     if confidence >= 0.85:
#         print(f"  {RED}⚠  HIGH RISK — Strong evidence of cardiac abnormality{RESET}")
#         print(f"  {RED}   Immediate clinical evaluation recommended{RESET}")
#     elif confidence >= 0.65:
#         print(f"  {YELLOW}⚠  MODERATE RISK — Possible cardiac abnormality{RESET}")
#         print(f"  {YELLOW}   Clinical review recommended{RESET}")
#     elif confidence >= 0.5:
#         print(f"  {YELLOW}ℹ  LOW-MODERATE RISK — Borderline result{RESET}")
#         print(f"  {YELLOW}   Follow-up advised{RESET}")
#     elif confidence >= 0.3:
#         print(f"  {GREEN}ℹ  LOW RISK — Likely normal ECG{RESET}")
#         print(f"  {GREEN}   Routine monitoring recommended{RESET}")
#     else:
#         print(f"  {GREEN}✓  VERY LOW RISK — Strong evidence of normal ECG{RESET}")
#         print(f"  {GREEN}   No immediate action required{RESET}")

#     print(f"  {'─'*54}")
#     print(f"\n  {YELLOW}⚕ DISCLAIMER: This is a research tool only.{RESET}")
#     print(f"  {YELLOW}  Not for clinical diagnosis. Consult a cardiologist.{RESET}")
#     print(f"{BOLD}{'═'*58}{RESET}\n")


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit_diagram(ecg_feats):
#     """
#     Draw quantum circuit using IBM Qiskit.
#     Also submits circuit to IBM Quantum backend if API key is configured.

#     To use IBM Quantum backend:
#       1. Run once to save account:
#          from qiskit_ibm_runtime import QiskitRuntimeService
#          QiskitRuntimeService.save_account(
#              channel="ibm_quantum_platform",
#              token="YOUR_API_KEY_HERE",
#              overwrite=True
#          )
#       2. Then run terminal_multimodal.py normally.
#          It will detect saved credentials automatically.
#     """

#     print(f"\n{CYAN}Drawing IBM Quantum Circuit …{RESET}")

#     try:
#         from qiskit import QuantumCircuit, transpile
#         import matplotlib.pyplot as plt
#     except ImportError:
#         print(f"{YELLOW}Qiskit not installed.{RESET}")
#         print(f"Run:  pip install qiskit qiskit-ibm-runtime pylatexenc")
#         _draw_pennylane_circuit(ecg_feats)
#         return

#     N_QUBITS = 8
#     N_LAYERS = 3   # anti-barren-plateau: 3 layers

#     # scale input features for angle encoding
#     sample = np.tanh(ecg_feats[:N_QUBITS]) * np.pi

#     # ── Build circuit ──────────────────────────────────────
#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)
#     qc.h(range(N_QUBITS))
#     qc.barrier()

#     for layer in range(N_LAYERS):
#         # Data re-uploading: RY encoding at every layer
#         for i in range(N_QUBITS):
#             angle = float(sample[i]) if i < len(sample) else 0.0
#             qc.ry(round(angle, 3), i)
#         qc.barrier()

#         # Variational rotations: RZ-RY-RZ per qubit
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)

#         # CNOT ring entanglement
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)
#         qc.barrier()

#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     # ── Print stats ────────────────────────────────────────
#     print(f"\n  {GREEN}IBM Quantum Circuit Stats:{RESET}")
#     print(f"  Qubits        : {N_QUBITS}")
#     print(f"  Layers        : {N_LAYERS}  (anti-barren-plateau design)")
#     print(f"  Circuit depth : {qc.depth()}")
#     print(f"  Total gates   : {sum(qc.count_ops().values())}")
#     print(f"  Gate types    : {dict(qc.count_ops())}")

#     # ── Draw and save locally ──────────────────────────────
#     save_path = "ibm_quantum_circuit.png"
#     try:
#         fig = qc.draw(
#             output = "mpl",
#             style  = "clifford",
#             fold   = 40,
#             scale  = 0.7,
#         )
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"\n  {GREEN}✓ Circuit diagram saved → {save_path}{RESET}")
#         plt.close()
#     except Exception as e:
#         print(f"  {YELLOW}mpl draw failed ({e}){RESET}")
#         print(qc.draw(output="text", fold=80))

#     # ── IBM Quantum Backend (optional) ────────────────────
#     print(f"\n  {CYAN}Submit to IBM Quantum backend? (y/n): {RESET}", end="")
#     ibm_run = input().strip().lower()

#     if ibm_run == "y":
#         _run_on_ibm_backend(qc, N_QUBITS)

#     return qc


# def _run_on_ibm_backend(qc, N_QUBITS):
#     try:
#         from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
#         from qiskit import transpile
#     except ImportError:
#         print(f"{YELLOW}Install: pip install qiskit-ibm-runtime{RESET}")
#         return

#     print(f"\n{CYAN}Connecting to IBM Quantum Platform ...{RESET}")

#     # load saved account
#     try:
#         service = QiskitRuntimeService(channel="ibm_quantum_platform")
#         print(f"  {GREEN}OK  IBM account loaded{RESET}")
#     except Exception as e:
#         print(f"  {YELLOW}No saved account found: {e}{RESET}")
#         print(f"  Save your account first:")
#         print(f"  {CYAN}from qiskit_ibm_runtime import QiskitRuntimeService")
#         print(f"  QiskitRuntimeService.save_account(")
#         print(f"      channel='ibm_quantum_platform',")
#         print(f"      token='YOUR_API_KEY',")
#         print(f"      overwrite=True){RESET}")
#         return

#     # list all backends
#     print(f"\n  Available backends:")
#     try:
#         all_b = service.backends()
#         for b in all_b:
#             try:
#                 st = b.status()
#                 print(f"    {b.name:25s}  operational={st.operational}  pending={st.pending_jobs}")
#             except Exception:
#                 print(f"    {b.name}")
#     except Exception as e:
#         print(f"  {YELLOW}Could not list backends: {e}{RESET}")
#         all_b = []

#     # pick best backend — no qiskit_ibm_provider needed
#     try:
#         real_backends = []
#         for b in service.backends():
#             try:
#                 st  = b.status()
#                 cfg = b.configuration()
#                 if st.operational and cfg.n_qubits >= N_QUBITS:
#                     real_backends.append((b, st.pending_jobs))
#             except Exception:
#                 continue

#         if real_backends:
#             real_backends.sort(key=lambda x: x[1])
#             backend = real_backends[0][0]
#             print(f"\n  {GREEN}Selected: {backend.name}  ({backend.configuration().n_qubits} qubits)  pending={real_backends[0][1]}{RESET}")
#         else:
#             backend = service.backends()[0]
#             print(f"\n  {YELLOW}Using: {backend.name} (first available){RESET}")

#     except Exception as e:
#         print(f"  {RED}Backend selection error: {e}{RESET}")
#         return

#     # transpile and submit
#     print(f"\n  Transpiling for {backend.name} ...")
#     try:
#         tc  = transpile(qc, backend=backend, optimization_level=1)
#         print(f"  {GREEN}OK  Transpiled depth: {tc.depth()}{RESET}")

#         print(f"  Submitting (shots=1024) ...")
#         sampler = Sampler(mode=backend)
#         job     = sampler.run([tc], shots=1024)
#         job_id  = job.job_id()

#         print(f"  {GREEN}OK  Job submitted!{RESET}")
#         print(f"  Job ID : {job_id}")
#         print(f"  Track  : https://quantum.ibm.com/jobs/{job_id}")

#         with open("ibm_quantum_results.txt", "w") as f:
#             f.write(f"QMM-CardioNet IBM Quantum Job\n")
#             f.write(f"Backend : {backend.name}\n")
#             f.write(f"Job ID  : {job_id}\n")
#             f.write(f"URL     : https://quantum.ibm.com/jobs/{job_id}\n")
#             f.write(f"Qubits  : {N_QUBITS}\n")
#             f.write(f"Depth   : {tc.depth()}\n")

#         print(f"  {GREEN}OK  Saved → ibm_quantum_results.txt{RESET}")
#         print(f"\n  {CYAN}View on IBM Cloud: https://quantum.ibm.com/jobs/{job_id}{RESET}")

#     except Exception as e:
#         print(f"  {RED}Job submission failed: {e}{RESET}")
#         print(f"  Check quota and account at quantum.cloud.ibm.com")

# def _draw_pennylane_circuit(ecg_feats):
#     """Fallback: print PennyLane circuit in terminal."""
#     try:
#         import pennylane as qml
#         from models.quantum_layer import circuit, N_QUBITS, N_LAYERS
#         sample  = np.tanh(ecg_feats[:N_QUBITS]).astype(np.float32)
#         weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#         print("\nPennyLane Circuit:")
#         print("=" * 50)
#         print(qml.draw(circuit)(
#             torch.tensor(sample),
#             torch.tensor(weights)
#         ))
#         print("=" * 50)
#     except Exception as e:
#         print(f"{YELLOW}Circuit draw failed: {e}{RESET}")


# # ═══════════════════════════════════════════════════════════
# #  MAIN LOOP
# # ═══════════════════════════════════════════════════════════
# def main():
#     print_banner()

#     # load model + scaler
#     model, ecg_dim   = load_model()
#     scaler, _        = get_scaler_from_cache()

#     while True:
#         try:
#             # get patient input
#             patient = get_patient_input()

#             # extract features
#             ecg_feats, clin_feats = extract_features(patient)

#             # predict
#             print(f"\n{CYAN}Running Quantum Multimodal inference …{RESET}")
#             t0         = time.time()
#             confidence = predict(model, ecg_feats, clin_feats, scaler, ecg_dim)
#             inf_time   = time.time() - t0
#             print(f"  {GREEN}✓ Inference done in {inf_time:.2f}s{RESET}")

#             # display result
#             display_result(patient, confidence)

#             # IBM circuit
#             if patient["draw_ibm"]:
#                 draw_ibm_circuit_diagram(ecg_feats)

#             # ask to run again
#             print(f"\n{'─'*58}")
#             again = input(f"  Test another patient? (y/n): ").strip().lower()
#             if again != "y":
#                 print(f"\n{GREEN}Thank you for using QMM CARDIONET2!{RESET}\n")
#                 break

#         except KeyboardInterrupt:
#             print(f"\n\n{YELLOW}Interrupted by user.{RESET}\n")
#             break
#         except Exception as e:
#             print(f"\n{RED}Error: {e}{RESET}")
#             import traceback; traceback.print_exc()
#             again = input("Try again? (y/n): ").strip().lower()
#             if again != "y":
#                 break


# if __name__ == "__main__":
#     main()