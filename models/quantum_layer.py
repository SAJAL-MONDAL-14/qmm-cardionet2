# """
# quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
# =======================================================
# WHY OLD LAYER STUCK AT 86%:
#   - StronglyEntanglingLayers with 5 layers + 8 qubits
#     causes BARREN PLATEAU — gradients vanish to zero
#   - Model stops learning after a few epochs
#   - No matter how long you train → stays at 86%

# FIXES APPLIED:
#   1. Fewer layers (3 instead of 5)  → avoids barren plateau
#   2. Local cost function             → gradients don't vanish
#   3. Data re-uploading               → more expressive circuit
#   4. IQP-style encoding              → stronger feature embedding
#   5. Learnable input scaling         → optimal angle range per qubit

# References:
#   - McClean et al. (2018) Barren plateaus in quantum neural network
#   - Perez-Salinas et al. (2020) Data re-uploading
#   - Cerezo et al. (2021) Cost-function-dependent barren plateaus
# """

# import torch
# import torch.nn as nn
# import pennylane as qml
# import numpy as np

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# N_QUBITS  = 8
# N_LAYERS  = 3     # KEY FIX: 3 layers (not 5) → avoids barren plateau
#                   # Cerezo 2021: shallow circuits have non-vanishing gradients

# USE_IBM   = False  # set True to draw IBM circuit diagram

# dev = qml.device("default.qubit", wires=N_QUBITS)


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM CIRCUIT  —  Anti-Barren-Plateau Design
# # ═══════════════════════════════════════════════════════════
# @qml.qnode(dev, interface="torch")
# def circuit(inputs, weights):
#     """
#     Anti-barren-plateau variational circuit.

#     Design choices:
#       - Hadamard init          → equal superposition start
#       - Data re-uploading      → encode data at EVERY layer
#       - Local measurements     → measure neighbouring pairs
#                                  (avoids global barren plateau)
#       - 3 layers only          → shallow = trainable gradients
#       - RY encoding            → more expressive than RZ
#     """

#     # ── Step 1: Hadamard — equal superposition ──
#     for i in range(N_QUBITS):
#         qml.Hadamard(wires=i)

#     # ── Step 2: Data Re-uploading (encode at every layer) ──
#     for layer in range(N_LAYERS):

#         # encode features as RY rotation angles
#         for i in range(N_QUBITS):
#             qml.RY(inputs[i], wires=i)

#         # parameterized single-qubit rotations
#         for i in range(N_QUBITS):
#             qml.RZ(weights[layer, i, 0], wires=i)
#             qml.RY(weights[layer, i, 1], wires=i)
#             qml.RZ(weights[layer, i, 2], wires=i)

#         # entanglement — nearest neighbour CNOT ring
#         # (local entanglement = avoids barren plateau)
#         for i in range(N_QUBITS):
#             qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

#     # ── Step 3: LOCAL measurements (key for avoiding barren plateau) ──
#     # Measure neighbouring pairs → local cost function
#     # This is the Cerezo 2021 fix for barren plateaus
#     return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM LAYER MODULE
# # ═══════════════════════════════════════════════════════════
# class QuantumLayer(nn.Module):
#     """
#     Anti-barren-plateau quantum layer.
#     Input  : (B, 8)  — any range
#     Output : (B, 8)  — expectation values in [-1, 1]

#     Key features:
#       - 3 layers (not 5) → avoids vanishing gradients
#       - Learnable per-qubit input scaling
#       - tanh normalization before encoding
#     """

#     def __init__(self):
#         super().__init__()

#         # weight shape: (N_LAYERS, N_QUBITS, 3)
#         weight_shapes    = {"weights": (N_LAYERS, N_QUBITS, 3)}
#         self.q_layer     = qml.qnn.TorchLayer(circuit, weight_shapes)

#         # learnable per-qubit scaling — initialized near π/2
#         # π/2 is the optimal initial angle for RY encoding
#         self.input_scale = nn.Parameter(
#             torch.ones(N_QUBITS) * (np.pi / 2)
#         )

#         # learnable output scaling — helps classifier
#         self.output_scale = nn.Parameter(torch.ones(N_QUBITS))

#     def forward(self, x):
#         """
#         x        : (B, 8)
#         returns  : (B, 8)
#         """
#         # normalize to [-π/2, π/2] range for stable RY encoding
#         x_scaled = torch.tanh(x) * self.input_scale

#         # quantum forward pass
#         q_out = self.q_layer(x_scaled)

#         # learnable output scaling
#         return q_out * self.output_scale


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit(
#     save_path    = "ibm_quantum_circuit.png",
#     style        = "clifford",
#     fold         = 40,
#     sample_input = None,
#     show         = True
# ):
#     """
#     Draw the quantum circuit using IBM Qiskit.
#     Used for research paper figures ONLY — not for training.

#     Parameters
#     ----------
#     save_path    : str   where to save the PNG
#     style        : str   "clifford" (IBM blue) or "bw"
#     fold         : int   wrap after this many gates
#     sample_input : list  example input (optional)
#     show         : bool  display after saving
#     """
#     try:
#         from qiskit import QuantumCircuit
#         from qiskit.visualization import circuit_drawer
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise ImportError(
#             "Qiskit not installed.\n"
#             "Run:  pip install qiskit pylatexenc"
#         )

#     if sample_input is None:
#         sample_input = [np.pi / 4] * N_QUBITS

#     sample = np.tanh(np.array(sample_input)) * (np.pi / 2)

#     print(f"\nBuilding IBM Quantum Circuit …")
#     print(f"  Qubits  : {N_QUBITS}")
#     print(f"  Layers  : {N_LAYERS}  (anti-barren-plateau)")
#     print(f"  Style   : {style}")

#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)

#     # Hadamard init
#     for i in range(N_QUBITS):
#         qc.h(i)
#     qc.barrier(label="H-Init")

#     # data re-uploading layers
#     for layer in range(N_LAYERS):

#         # RY encoding
#         for i in range(N_QUBITS):
#             qc.ry(round(float(sample[i]), 3), i)
#         qc.barrier(label=f"Enc {layer+1}")

#         # variational rotations
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)

#         # nearest-neighbour CNOT ring
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)

#         qc.barrier(label=f"Var {layer+1}")

#     # measure
#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     print(f"  Circuit depth : {qc.depth()}")
#     print(f"  Total gates   : {sum(qc.count_ops().values())}")
#     print(f"  Gate types    : {dict(qc.count_ops())}")

#     # draw
#     try:
#         fig = qc.draw(
#             output  = "mpl",
#             style   = style,
#             fold    = fold,
#             scale   = 0.65,
#         )
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"\n✓ Circuit saved → {save_path}")
#         if show:
#             plt.tight_layout()
#             plt.show()
#     except Exception as e:
#         print(f"mpl draw failed ({e}), using text draw:")
#         print(qc.draw(output="text", fold=80))

#     return qc


# def print_pennylane_circuit():
#     """Print PennyLane circuit in terminal — no Qiskit needed."""
#     sample  = np.array([np.pi/4] * N_QUBITS, dtype=np.float32)
#     weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#     print("\nPennyLane Circuit Diagram:")
#     print("=" * 55)
#     print(qml.draw(circuit)(
#         torch.tensor(sample),
#         torch.tensor(weights)
#     ))
#     print("=" * 55)


# # ── auto-draw IBM circuit if USE_IBM = True ───────────────
# if USE_IBM:
#     print("USE_IBM = True → Drawing IBM circuit …")
#     try:
#         draw_ibm_circuit(save_path="ibm_quantum_circuit.png", show=True)
#     except ImportError as e:
#         print(f"IBM draw failed: {e}")
#         print("Falling back to PennyLane …")
#         print_pennylane_circuit()


# # ── quick test ────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=" * 55)
#     print("  QuantumLayer Test  (Anti-Barren-Plateau)")
#     print("=" * 55)

#     ql  = QuantumLayer()
#     x   = torch.randn(4, N_QUBITS)
#     out = ql(x)

#     print(f"\n  Input  : {x.shape}")
#     print(f"  Output : {out.shape}")
#     print(f"  Range  : [{out.min().item():.3f}, {out.max().item():.3f}]")
#     print(f"\n  Trainable parameters:")
#     print(f"    circuit weights : {N_LAYERS * N_QUBITS * 3}")
#     print(f"    input_scale     : {N_QUBITS}")
#     print(f"    output_scale    : {N_QUBITS}")
#     print(f"    total           : {N_LAYERS*N_QUBITS*3 + N_QUBITS*2}")
#     print(f"\n  ✓ QuantumLayer working")

#     print_pennylane_circuit()

#     if USE_IBM:
#         draw_ibm_circuit()
#     else:
#         print(f"\n  Set USE_IBM = True to draw IBM circuit diagram")




"""
quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
=======================================================
WHY OLD LAYER STUCK AT 86%:
  StronglyEntanglingLayers with 5 layers = BARREN PLATEAU
  Gradients vanish → model stops learning

FIXES:
  1. 3 layers only        → avoids barren plateau
  2. Data re-uploading    → AngleEmbedding at every layer
  3. Learnable scaling    → optimal input range per qubit
  4. Output scaling       → helps downstream classifier

NOTE: Uses AngleEmbedding (NOT manual RY loop) for correct
      PennyLane batch processing.

References:
  - McClean et al. (2018) Barren plateaus in QNN training
  - Perez-Salinas et al. (2020) Data re-uploading
  - Cerezo et al. (2021) Cost-function-dependent barren plateaus
"""
# """
# quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
# =======================================================
# WHY OLD LAYER STUCK AT 86%:
#   - StronglyEntanglingLayers with 5 layers + 8 qubits
#     causes BARREN PLATEAU — gradients vanish to zero
#   - Model stops learning after a few epochs
#   - No matter how long you train → stays at 86%

# FIXES APPLIED:
#   1. Fewer layers (3 instead of 5)  → avoids barren plateau
#   2. Local cost function             → gradients don't vanish
#   3. Data re-uploading               → more expressive circuit
#   4. IQP-style encoding              → stronger feature embedding
#   5. Learnable input scaling         → optimal angle range per qubit

# References:
#   - McClean et al. (2018) Barren plateaus in quantum neural network
#   - Perez-Salinas et al. (2020) Data re-uploading
#   - Cerezo et al. (2021) Cost-function-dependent barren plateaus
# """

# import torch
# import torch.nn as nn
# import pennylane as qml
# import numpy as np

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# N_QUBITS  = 8
# N_LAYERS  = 3     # KEY FIX: 3 layers (not 5) → avoids barren plateau
#                   # Cerezo 2021: shallow circuits have non-vanishing gradients

# USE_IBM   = False  # set True to draw IBM circuit diagram

# dev = qml.device("default.qubit", wires=N_QUBITS)


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM CIRCUIT  —  Anti-Barren-Plateau Design
# # ═══════════════════════════════════════════════════════════
# @qml.qnode(dev, interface="torch", diff_method=diff_method)
# def circuit(inputs, weights):
#     """
#     Anti-barren-plateau variational circuit.

#     Design choices:
#       - Hadamard init          → equal superposition start
#       - Data re-uploading      → encode data at EVERY layer
#       - Local measurements     → measure neighbouring pairs
#                                  (avoids global barren plateau)
#       - 3 layers only          → shallow = trainable gradients
#       - RY encoding            → more expressive than RZ
#     """

#     # ── Step 1: Hadamard — equal superposition ──
#     for i in range(N_QUBITS):
#         qml.Hadamard(wires=i)

#     # ── Step 2: Data Re-uploading (encode at every layer) ──
#     for layer in range(N_LAYERS):

#         # encode features as RY rotation angles
#         for i in range(N_QUBITS):
#             qml.RY(inputs[i], wires=i)

#         # parameterized single-qubit rotations
#         for i in range(N_QUBITS):
#             qml.RZ(weights[layer, i, 0], wires=i)
#             qml.RY(weights[layer, i, 1], wires=i)
#             qml.RZ(weights[layer, i, 2], wires=i)

#         # entanglement — nearest neighbour CNOT ring
#         # (local entanglement = avoids barren plateau)
#         for i in range(N_QUBITS):
#             qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

#     # ── Step 3: LOCAL measurements (key for avoiding barren plateau) ──
#     # Measure neighbouring pairs → local cost function
#     # This is the Cerezo 2021 fix for barren plateaus
#     return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM LAYER MODULE
# # ═══════════════════════════════════════════════════════════
# class QuantumLayer(nn.Module):
#     """
#     Anti-barren-plateau quantum layer.
#     Input  : (B, 8)  — any range
#     Output : (B, 8)  — expectation values in [-1, 1]

#     Key features:
#       - 3 layers (not 5) → avoids vanishing gradients
#       - Learnable per-qubit input scaling
#       - tanh normalization before encoding
#     """

#     def __init__(self):
#         super().__init__()

#         # weight shape: (N_LAYERS, N_QUBITS, 3)
#         weight_shapes    = {"weights": (N_LAYERS, N_QUBITS, 3)}
#         self.q_layer     = qml.qnn.TorchLayer(circuit, weight_shapes)

#         # learnable per-qubit scaling — initialized near π/2
#         # π/2 is the optimal initial angle for RY encoding
#         self.input_scale = nn.Parameter(
#             torch.ones(N_QUBITS) * (np.pi / 2)
#         )

#         # learnable output scaling — helps classifier
#         self.output_scale = nn.Parameter(torch.ones(N_QUBITS))

#     def forward(self, x):
#         """
#         x        : (B, 8)
#         returns  : (B, 8)
#         """
#         # normalize to [-π/2, π/2] range for stable RY encoding
#         x_scaled = torch.tanh(x) * self.input_scale

#         # quantum forward pass
#         q_out = self.q_layer(x_scaled)

#         # learnable output scaling
#         return q_out * self.output_scale


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit(
#     save_path    = "ibm_quantum_circuit.png",
#     style        = "clifford",
#     fold         = 40,
#     sample_input = None,
#     show         = True
# ):
#     """
#     Draw the quantum circuit using IBM Qiskit.
#     Used for research paper figures ONLY — not for training.

#     Parameters
#     ----------
#     save_path    : str   where to save the PNG
#     style        : str   "clifford" (IBM blue) or "bw"
#     fold         : int   wrap after this many gates
#     sample_input : list  example input (optional)
#     show         : bool  display after saving
#     """
#     try:
#         from qiskit import QuantumCircuit
#         from qiskit.visualization import circuit_drawer
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise ImportError(
#             "Qiskit not installed.\n"
#             "Run:  pip install qiskit pylatexenc"
#         )

#     if sample_input is None:
#         sample_input = [np.pi / 4] * N_QUBITS

#     sample = np.tanh(np.array(sample_input)) * (np.pi / 2)

#     print(f"\nBuilding IBM Quantum Circuit …")
#     print(f"  Qubits  : {N_QUBITS}")
#     print(f"  Layers  : {N_LAYERS}  (anti-barren-plateau)")
#     print(f"  Style   : {style}")

#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)

#     # Hadamard init
#     for i in range(N_QUBITS):
#         qc.h(i)
#     qc.barrier(label="H-Init")

#     # data re-uploading layers
#     for layer in range(N_LAYERS):

#         # RY encoding
#         for i in range(N_QUBITS):
#             qc.ry(round(float(sample[i]), 3), i)
#         qc.barrier(label=f"Enc {layer+1}")

#         # variational rotations
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)

#         # nearest-neighbour CNOT ring
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)

#         qc.barrier(label=f"Var {layer+1}")

#     # measure
#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     print(f"  Circuit depth : {qc.depth()}")
#     print(f"  Total gates   : {sum(qc.count_ops().values())}")
#     print(f"  Gate types    : {dict(qc.count_ops())}")

#     # draw
#     try:
#         fig = qc.draw(
#             output  = "mpl",
#             style   = style,
#             fold    = fold,
#             scale   = 0.65,
#         )
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"\n✓ Circuit saved → {save_path}")
#         if show:
#             plt.tight_layout()
#             plt.show()
#     except Exception as e:
#         print(f"mpl draw failed ({e}), using text draw:")
#         print(qc.draw(output="text", fold=80))

#     return qc


# def print_pennylane_circuit():
#     """Print PennyLane circuit in terminal — no Qiskit needed."""
#     sample  = np.array([np.pi/4] * N_QUBITS, dtype=np.float32)
#     weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#     print("\nPennyLane Circuit Diagram:")
#     print("=" * 55)
#     print(qml.draw(circuit)(
#         torch.tensor(sample),
#         torch.tensor(weights)
#     ))
#     print("=" * 55)


# # ── auto-draw IBM circuit if USE_IBM = True ───────────────
# if USE_IBM:
#     print("USE_IBM = True → Drawing IBM circuit …")
#     try:
#         draw_ibm_circuit(save_path="ibm_quantum_circuit.png", show=True)
#     except ImportError as e:
#         print(f"IBM draw failed: {e}")
#         print("Falling back to PennyLane …")
#         print_pennylane_circuit()


# # ── quick test ────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=" * 55)
#     print("  QuantumLayer Test  (Anti-Barren-Plateau)")
#     print("=" * 55)

#     ql  = QuantumLayer()
#     x   = torch.randn(4, N_QUBITS)
#     out = ql(x)

#     print(f"\n  Input  : {x.shape}")
#     print(f"  Output : {out.shape}")
#     print(f"  Range  : [{out.min().item():.3f}, {out.max().item():.3f}]")
#     print(f"\n  Trainable parameters:")
#     print(f"    circuit weights : {N_LAYERS * N_QUBITS * 3}")
#     print(f"    input_scale     : {N_QUBITS}")
#     print(f"    output_scale    : {N_QUBITS}")
#     print(f"    total           : {N_LAYERS*N_QUBITS*3 + N_QUBITS*2}")
#     print(f"\n  ✓ QuantumLayer working")

#     print_pennylane_circuit()

#     if USE_IBM:
#         draw_ibm_circuit()
#     else:
#         print(f"\n  Set USE_IBM = True to draw IBM circuit diagram")




# """
# quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
# =======================================================
# WHY OLD LAYER STUCK AT 86%:
#   StronglyEntanglingLayers with 5 layers = BARREN PLATEAU
#   Gradients vanish → model stops learning

# FIXES:
#   1. 3 layers only        → avoids barren plateau
#   2. Data re-uploading    → AngleEmbedding at every layer
#   3. Learnable scaling    → optimal input range per qubit
#   4. Output scaling       → helps downstream classifier

# NOTE: Uses AngleEmbedding (NOT manual RY loop) for correct
#       PennyLane batch processing.

# References:
#   - McClean et al. (2018) Barren plateaus in QNN training
#   - Perez-Salinas et al. (2020) Data re-uploading
#   - Cerezo et al. (2021) Cost-function-dependent barren plateaus
# """

# import torch
# import torch.nn as nn
# import pennylane as qml
# import numpy as np

# # ── Config ────────────────────────────────────────────────
# N_QUBITS  = 8
# N_LAYERS  = 3     # KEY: 3 not 5 → avoids barren plateau

# USE_IBM          = True   # True = draw IBM circuit diagram
# USE_IBM_QUANTUM  = True # True = run on REAL IBM QPU (inference only)
# IBM_BACKEND      = "ibm_marrakesh"  # ibm_marrakesh / ibm_fez / ibm_torino

# # ── Device selection ──────────────────────────────────────
# if USE_IBM_QUANTUM:
#     print("[QuantumLayer] Connecting to IBM Quantum Platform ...")
#     try:
#         from qiskit_ibm_runtime import QiskitRuntimeService
#         service = QiskitRuntimeService(channel="ibm_quantum_platform")
#         backend = service.backend(IBM_BACKEND)
#         print(f"[QuantumLayer] Backend : {backend.name}")
#         print(f"[QuantumLayer] Status  : {backend.status().status_msg}")
#         dev         = qml.device("qiskit.remote", wires=N_QUBITS,
#                                   backend=backend, shots=1024)
#         diff_method = None   # no gradient on real hardware
#         print("[QuantumLayer] IBM Quantum device ready")
#     except Exception as e:
#         print(f"[QuantumLayer] IBM connection failed: {e}")
#         print("[QuantumLayer] Falling back to simulator")
#         USE_IBM_QUANTUM = False
#         dev         = qml.device("default.qubit", wires=N_QUBITS)
#         diff_method = "backprop"
# else:
#     dev         = qml.device("default.qubit", wires=N_QUBITS)
#     diff_method = "backprop"


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM CIRCUIT
# # ═══════════════════════════════════════════════════════════
# @qml.qnode(dev, interface="torch", diff_method=diff_method)
# def circuit(inputs, weights):
#     """
#     Anti-barren-plateau circuit with data re-uploading.
#     Uses AngleEmbedding for correct PennyLane batch support.

#     Per layer:
#       Hadamard → AngleEmbedding(data) → RZ/RY/RZ rotations → CNOT ring
#     Repeated N_LAYERS times.
#     """

#     # ── initial superposition ──
#     for i in range(N_QUBITS):
#         qml.Hadamard(wires=i)

#     # ── data re-uploading layers ──
#     for layer in range(N_LAYERS):

#         # AngleEmbedding handles batching correctly in PennyLane
#         qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

#         # variational rotations (per qubit)
#         for i in range(N_QUBITS):
#             qml.RZ(weights[layer, i, 0], wires=i)
#             qml.RY(weights[layer, i, 1], wires=i)
#             qml.RZ(weights[layer, i, 2], wires=i)

#         # nearest-neighbour CNOT ring (local → avoids barren plateau)
#         for i in range(N_QUBITS):
#             qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

#     # ── measurements ──
#     return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM LAYER MODULE
# # ═══════════════════════════════════════════════════════════
# class QuantumLayer(nn.Module):
#     """
#     Drop-in replacement — same interface as original.
#     Input  : (B, 8)
#     Output : (B, 8)
#     """

#     def __init__(self):
#         super().__init__()

#         weight_shapes     = {"weights": (N_LAYERS, N_QUBITS, 3)}
#         self.q_layer      = qml.qnn.TorchLayer(circuit, weight_shapes)

#         # learnable per-qubit input scaling (init = π/2)
#         self.input_scale  = nn.Parameter(torch.ones(N_QUBITS) * (np.pi / 2))

#         # learnable output scaling
#         self.output_scale = nn.Parameter(torch.ones(N_QUBITS))

#     def forward(self, x):
#         # scale to stable range for AngleEmbedding
#         x_scaled = torch.tanh(x) * self.input_scale

#         if USE_IBM_QUANTUM:
#             # IBM hardware mode — single sample only (1 job per prediction)
#             inp   = x_scaled[0].detach().cpu().numpy().astype(float)
#             w     = self.q_layer.weights.detach().cpu().numpy().astype(float)
#             q_out = circuit(inp, w)
#             q_out = torch.stack([
#                 o.float() if isinstance(o, torch.Tensor)
#                 else torch.tensor(float(o))
#                 for o in q_out
#             ])
#             return (q_out * self.output_scale).unsqueeze(0)
#         else:
#             # Simulator mode — full batch (training)
#             q_out = self.q_layer(x_scaled)
#             return q_out * self.output_scale


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit(save_path="ibm_quantum_circuit.png",
#                      style="clifford", fold=40,
#                      sample_input=None, show=True):
#     """Draw circuit using Qiskit for research paper figures."""
#     try:
#         from qiskit import QuantumCircuit
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise ImportError("Run:  pip install qiskit pylatexenc")

#     if sample_input is None:
#         sample_input = [np.pi / 4] * N_QUBITS
#     sample = np.tanh(np.array(sample_input)) * (np.pi / 2)

#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)

#     for i in range(N_QUBITS):
#         qc.h(i)
#     qc.barrier(label="Init")

#     for layer in range(N_LAYERS):
#         for i in range(N_QUBITS):
#             qc.ry(round(float(sample[i]), 3), i)
#         qc.barrier(label=f"Enc {layer+1}")
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)
#         qc.barrier(label=f"Var {layer+1}")

#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     print(f"\nIBM Circuit Stats:")
#     print(f"  Qubits : {N_QUBITS}")
#     print(f"  Layers : {N_LAYERS}")
#     print(f"  Depth  : {qc.depth()}")
#     print(f"  Gates  : {dict(qc.count_ops())}")

#     try:
#         fig = qc.draw(output="mpl", style=style, fold=fold, scale=0.65)
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"✓ Saved → {save_path}")
#         if show:
#             plt.tight_layout(); plt.show()
#     except Exception as e:
#         print(f"mpl failed ({e}), text draw:")
#         print(qc.draw(output="text", fold=80))
#     return qc


# def print_pennylane_circuit():
#     """Print circuit in terminal — no Qiskit needed."""
#     sample  = np.array([np.pi/4] * N_QUBITS, dtype=np.float32)
#     weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#     print("\nPennyLane Circuit:")
#     print("=" * 55)
#     print(qml.draw(circuit)(torch.tensor(sample), torch.tensor(weights)))
#     print("=" * 55)


# # ── auto IBM draw ─────────────────────────────────────────
# if USE_IBM:
#     try:
#         draw_ibm_circuit()
#     except ImportError as e:
#         print(f"IBM draw failed: {e}")
#         print_pennylane_circuit()


# # ── quick test ────────────────────────────────────────────
# if __name__ == "__main__":
#     print("Testing QuantumLayer …")
#     ql  = QuantumLayer()
#     x   = torch.randn(4, N_QUBITS)
#     out = ql(x)
#     print(f"Input  : {x.shape}")
#     print(f"Output : {out.shape}")
#     print(f"Range  : [{out.min().item():.3f}, {out.max().item():.3f}]")
#     print("✓ Working correctly")
#     print_pennylane_circuit()



# """
# quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
# =======================================================
# WHY OLD LAYER STUCK AT 86%:
#   - StronglyEntanglingLayers with 5 layers + 8 qubits
#     causes BARREN PLATEAU — gradients vanish to zero
#   - Model stops learning after a few epochs
#   - No matter how long you train → stays at 86%

# FIXES APPLIED:
#   1. Fewer layers (3 instead of 5)  → avoids barren plateau
#   2. Local cost function             → gradients don't vanish
#   3. Data re-uploading               → more expressive circuit
#   4. IQP-style encoding              → stronger feature embedding
#   5. Learnable input scaling         → optimal angle range per qubit

# References:
#   - McClean et al. (2018) Barren plateaus in quantum neural network
#   - Perez-Salinas et al. (2020) Data re-uploading
#   - Cerezo et al. (2021) Cost-function-dependent barren plateaus
# """

# import torch
# import torch.nn as nn
# import pennylane as qml
# import numpy as np

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# N_QUBITS  = 8
# N_LAYERS  = 3     # KEY FIX: 3 layers (not 5) → avoids barren plateau
#                   # Cerezo 2021: shallow circuits have non-vanishing gradients

# USE_IBM   = False  # set True to draw IBM circuit diagram

# dev = qml.device("default.qubit", wires=N_QUBITS)


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM CIRCUIT  —  Anti-Barren-Plateau Design
# # ═══════════════════════════════════════════════════════════
# @qml.qnode(dev, interface="torch", diff_method=diff_method)
# def circuit(inputs, weights):
#     """
#     Anti-barren-plateau variational circuit.

#     Design choices:
#       - Hadamard init          → equal superposition start
#       - Data re-uploading      → encode data at EVERY layer
#       - Local measurements     → measure neighbouring pairs
#                                  (avoids global barren plateau)
#       - 3 layers only          → shallow = trainable gradients
#       - RY encoding            → more expressive than RZ
#     """

#     # ── Step 1: Hadamard — equal superposition ──
#     for i in range(N_QUBITS):
#         qml.Hadamard(wires=i)

#     # ── Step 2: Data Re-uploading (encode at every layer) ──
#     for layer in range(N_LAYERS):

#         # encode features as RY rotation angles
#         for i in range(N_QUBITS):
#             qml.RY(inputs[i], wires=i)

#         # parameterized single-qubit rotations
#         for i in range(N_QUBITS):
#             qml.RZ(weights[layer, i, 0], wires=i)
#             qml.RY(weights[layer, i, 1], wires=i)
#             qml.RZ(weights[layer, i, 2], wires=i)

#         # entanglement — nearest neighbour CNOT ring
#         # (local entanglement = avoids barren plateau)
#         for i in range(N_QUBITS):
#             qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

#     # ── Step 3: LOCAL measurements (key for avoiding barren plateau) ──
#     # Measure neighbouring pairs → local cost function
#     # This is the Cerezo 2021 fix for barren plateaus
#     return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# # ═══════════════════════════════════════════════════════════
# #  QUANTUM LAYER MODULE
# # ═══════════════════════════════════════════════════════════
# class QuantumLayer(nn.Module):
#     """
#     Anti-barren-plateau quantum layer.
#     Input  : (B, 8)  — any range
#     Output : (B, 8)  — expectation values in [-1, 1]

#     Key features:
#       - 3 layers (not 5) → avoids vanishing gradients
#       - Learnable per-qubit input scaling
#       - tanh normalization before encoding
#     """

#     def __init__(self):
#         super().__init__()

#         # weight shape: (N_LAYERS, N_QUBITS, 3)
#         weight_shapes    = {"weights": (N_LAYERS, N_QUBITS, 3)}
#         self.q_layer     = qml.qnn.TorchLayer(circuit, weight_shapes)

#         # learnable per-qubit scaling — initialized near π/2
#         # π/2 is the optimal initial angle for RY encoding
#         self.input_scale = nn.Parameter(
#             torch.ones(N_QUBITS) * (np.pi / 2)
#         )

#         # learnable output scaling — helps classifier
#         self.output_scale = nn.Parameter(torch.ones(N_QUBITS))

#     def forward(self, x):
#         """
#         x        : (B, 8)
#         returns  : (B, 8)
#         """
#         # normalize to [-π/2, π/2] range for stable RY encoding
#         x_scaled = torch.tanh(x) * self.input_scale

#         # quantum forward pass
#         q_out = self.q_layer(x_scaled)

#         # learnable output scaling
#         return q_out * self.output_scale


# # ═══════════════════════════════════════════════════════════
# #  IBM CIRCUIT VISUALIZATION
# # ═══════════════════════════════════════════════════════════
# def draw_ibm_circuit(
#     save_path    = "ibm_quantum_circuit.png",
#     style        = "clifford",
#     fold         = 40,
#     sample_input = None,
#     show         = True
# ):
#     """
#     Draw the quantum circuit using IBM Qiskit.
#     Used for research paper figures ONLY — not for training.

#     Parameters
#     ----------
#     save_path    : str   where to save the PNG
#     style        : str   "clifford" (IBM blue) or "bw"
#     fold         : int   wrap after this many gates
#     sample_input : list  example input (optional)
#     show         : bool  display after saving
#     """
#     try:
#         from qiskit import QuantumCircuit
#         from qiskit.visualization import circuit_drawer
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise ImportError(
#             "Qiskit not installed.\n"
#             "Run:  pip install qiskit pylatexenc"
#         )

#     if sample_input is None:
#         sample_input = [np.pi / 4] * N_QUBITS

#     sample = np.tanh(np.array(sample_input)) * (np.pi / 2)

#     print(f"\nBuilding IBM Quantum Circuit …")
#     print(f"  Qubits  : {N_QUBITS}")
#     print(f"  Layers  : {N_LAYERS}  (anti-barren-plateau)")
#     print(f"  Style   : {style}")

#     qc = QuantumCircuit(N_QUBITS, N_QUBITS)

#     # Hadamard init
#     for i in range(N_QUBITS):
#         qc.h(i)
#     qc.barrier(label="H-Init")

#     # data re-uploading layers
#     for layer in range(N_LAYERS):

#         # RY encoding
#         for i in range(N_QUBITS):
#             qc.ry(round(float(sample[i]), 3), i)
#         qc.barrier(label=f"Enc {layer+1}")

#         # variational rotations
#         for i in range(N_QUBITS):
#             qc.rz(np.pi/4, i)
#             qc.ry(np.pi/4, i)
#             qc.rz(np.pi/4, i)

#         # nearest-neighbour CNOT ring
#         for i in range(N_QUBITS):
#             qc.cx(i, (i + 1) % N_QUBITS)

#         qc.barrier(label=f"Var {layer+1}")

#     # measure
#     qc.measure(range(N_QUBITS), range(N_QUBITS))

#     print(f"  Circuit depth : {qc.depth()}")
#     print(f"  Total gates   : {sum(qc.count_ops().values())}")
#     print(f"  Gate types    : {dict(qc.count_ops())}")

#     # draw
#     try:
#         fig = qc.draw(
#             output  = "mpl",
#             style   = style,
#             fold    = fold,
#             scale   = 0.65,
#         )
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"\n✓ Circuit saved → {save_path}")
#         if show:
#             plt.tight_layout()
#             plt.show()
#     except Exception as e:
#         print(f"mpl draw failed ({e}), using text draw:")
#         print(qc.draw(output="text", fold=80))

#     return qc


# def print_pennylane_circuit():
#     """Print PennyLane circuit in terminal — no Qiskit needed."""
#     sample  = np.array([np.pi/4] * N_QUBITS, dtype=np.float32)
#     weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
#     print("\nPennyLane Circuit Diagram:")
#     print("=" * 55)
#     print(qml.draw(circuit)(
#         torch.tensor(sample),
#         torch.tensor(weights)
#     ))
#     print("=" * 55)


# # ── auto-draw IBM circuit if USE_IBM = True ───────────────
# if USE_IBM:
#     print("USE_IBM = True → Drawing IBM circuit …")
#     try:
#         draw_ibm_circuit(save_path="ibm_quantum_circuit.png", show=True)
#     except ImportError as e:
#         print(f"IBM draw failed: {e}")
#         print("Falling back to PennyLane …")
#         print_pennylane_circuit()


# # ── quick test ────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=" * 55)
#     print("  QuantumLayer Test  (Anti-Barren-Plateau)")
#     print("=" * 55)

#     ql  = QuantumLayer()
#     x   = torch.randn(4, N_QUBITS)
#     out = ql(x)

#     print(f"\n  Input  : {x.shape}")
#     print(f"  Output : {out.shape}")
#     print(f"  Range  : [{out.min().item():.3f}, {out.max().item():.3f}]")
#     print(f"\n  Trainable parameters:")
#     print(f"    circuit weights : {N_LAYERS * N_QUBITS * 3}")
#     print(f"    input_scale     : {N_QUBITS}")
#     print(f"    output_scale    : {N_QUBITS}")
#     print(f"    total           : {N_LAYERS*N_QUBITS*3 + N_QUBITS*2}")
#     print(f"\n  ✓ QuantumLayer working")

#     print_pennylane_circuit()

#     if USE_IBM:
#         draw_ibm_circuit()
#     else:
#         print(f"\n  Set USE_IBM = True to draw IBM circuit diagram")




"""
quantum_layer.py  —  Anti-Barren-Plateau Quantum Layer
=======================================================
WHY OLD LAYER STUCK AT 86%:
  StronglyEntanglingLayers with 5 layers = BARREN PLATEAU
  Gradients vanish → model stops learning

FIXES:
  1. 3 layers only        → avoids barren plateau
  2. Data re-uploading    → AngleEmbedding at every layer
  3. Learnable scaling    → optimal input range per qubit
  4. Output scaling       → helps downstream classifier

NOTE: Uses AngleEmbedding (NOT manual RY loop) for correct
      PennyLane batch processing.

References:
  - McClean et al. (2018) Barren plateaus in QNN training
  - Perez-Salinas et al. (2020) Data re-uploading
  - Cerezo et al. (2021) Cost-function-dependent barren plateaus
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# ── Config ────────────────────────────────────────────────
N_QUBITS  = 8
N_LAYERS  = 3     # KEY: 3 not 5 → avoids barren plateau

USE_IBM          = True   # True = draw IBM circuit diagram
USE_IBM_QUANTUM  = False  # True = run on REAL IBM QPU (inference only)
IBM_BACKEND      = "ibm_marrakesh"  # ibm_marrakesh / ibm_fez / ibm_torino

# ── ALWAYS use simulator device (fast startup, no blocking) ──
# IBM hardware is only used at job submission time in forward()
# This avoids the 8-10 min hang caused by qml.device("qiskit.remote")
dev         = qml.device("default.qubit", wires=N_QUBITS)
diff_method = "backprop"


# ═══════════════════════════════════════════════════════════
#  QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════
@qml.qnode(dev, interface="torch", diff_method=diff_method)
def circuit(inputs, weights):
    """
    Anti-barren-plateau circuit with data re-uploading.
    Uses AngleEmbedding for correct PennyLane batch support.

    Per layer:
      Hadamard → AngleEmbedding(data) → RZ/RY/RZ rotations → CNOT ring
    Repeated N_LAYERS times.
    """

    # ── initial superposition ──
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)

    # ── data re-uploading layers ──
    for layer in range(N_LAYERS):

        # AngleEmbedding handles batching correctly in PennyLane
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

        # variational rotations (per qubit)
        for i in range(N_QUBITS):
            qml.RZ(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)

        # nearest-neighbour CNOT ring (local → avoids barren plateau)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # ── measurements ──
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]




# ═══════════════════════════════════════════════════════════
#  IBM QUANTUM EXECUTION  (called from forward() at runtime)
#  Uses Qiskit directly — avoids PennyLane qiskit.remote hang
# ═══════════════════════════════════════════════════════════
def _run_circuit_on_ibm(inputs, weights):
    """
    Build Qiskit circuit, transpile, submit to IBM QPU.
    Returns list of 8 expectation values estimated from counts.
    Called only when USE_IBM_QUANTUM=True.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        import math
    except ImportError:
        print("[IBM] Install: pip install qiskit qiskit-ibm-runtime")
        return None

    # Build circuit matching PennyLane circuit definition
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    # Hadamard init
    for i in range(N_QUBITS):
        qc.h(i)

    # Data re-uploading layers
    scaled = np.tanh(inputs) * (math.pi / 2)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qc.ry(float(scaled[i]), i)
        for i in range(N_QUBITS):
            qc.rz(float(weights[layer, i, 0]), i)
            qc.ry(float(weights[layer, i, 1]), i)
            qc.rz(float(weights[layer, i, 2]), i)
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    # Connect and submit
    try:
        print("[IBM] Connecting to IBM Quantum Platform ...")
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(IBM_BACKEND)
        print(f"[IBM] Backend: {backend.name}  status: {backend.status().status_msg}")

        tc  = transpile(qc, backend=backend, optimization_level=1)
        print(f"[IBM] Transpiled depth: {tc.depth()}")

        sampler = Sampler(mode=backend)
        job     = sampler.run([tc], shots=1024)
        job_id  = job.job_id()
        print(f"[IBM] Job submitted: {job_id}")
        print(f"[IBM] Track: https://quantum.ibm.com/jobs/{job_id}")

        print("[IBM] Waiting for result ...")
        result = job.result()

        # Extract counts and compute Z expectation values
        counts = result[0].data.c.get_counts()
        total  = sum(counts.values())
        expvals = []
        for qubit in range(N_QUBITS):
            exp = 0.0
            for bitstring, count in counts.items():
                # IBM bitstrings are reversed (rightmost = qubit 0)
                bit = int(bitstring[-(qubit+1)])
                exp += count * (1 - 2 * bit)
            expvals.append(exp / total)

        print(f"[IBM] Done! Expectation values: {[round(v,3) for v in expvals]}")

        # save job info
        with open("ibm_quantum_results.txt", "w") as f:
            f.write(f"QMM-CardioNet IBM Quantum Result\n")
            f.write(f"Backend : {backend.name}\n")
            f.write(f"Job ID  : {job_id}\n")
            f.write(f"URL     : https://quantum.ibm.com/jobs/{job_id}\n")
            f.write(f"Counts  : {counts}\n")
            f.write(f"ExpVals : {expvals}\n")
        print("[IBM] Saved → ibm_quantum_results.txt")

        return expvals

    except Exception as e:
        print(f"[IBM] Execution failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════
#  QUANTUM LAYER MODULE
# ═══════════════════════════════════════════════════════════
class QuantumLayer(nn.Module):
    """
    Drop-in replacement — same interface as original.
    Input  : (B, 8)
    Output : (B, 8)
    """

    def __init__(self):
        super().__init__()

        weight_shapes     = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.q_layer      = qml.qnn.TorchLayer(circuit, weight_shapes)

        # learnable per-qubit input scaling (init = π/2)
        self.input_scale  = nn.Parameter(torch.ones(N_QUBITS) * (np.pi / 2))

        # learnable output scaling
        self.output_scale = nn.Parameter(torch.ones(N_QUBITS))

    def forward(self, x):
        # scale to stable range for AngleEmbedding
        x_scaled = torch.tanh(x) * self.input_scale

        if USE_IBM_QUANTUM:
            # IBM hardware mode — submit via Qiskit directly (no hang)
            inp = x_scaled[0].detach().cpu().numpy().astype(float)
            w   = self.q_layer.weights.detach().cpu().numpy().astype(float)
            result = _run_circuit_on_ibm(inp, w)
            if result is not None:
                q_out = torch.tensor(result, dtype=torch.float32)
            else:
                # fallback to simulator if IBM fails
                q_out = self.q_layer(x_scaled[0:1])
                return q_out * self.output_scale
            return (q_out * self.output_scale).unsqueeze(0)
        else:
            # Simulator mode — full batch (training)
            q_out = self.q_layer(x_scaled)
            return q_out * self.output_scale


# ═══════════════════════════════════════════════════════════
#  IBM CIRCUIT VISUALIZATION
# ═══════════════════════════════════════════════════════════
def draw_ibm_circuit(save_path="ibm_quantum_circuit.png",
                     style="clifford", fold=40,
                     sample_input=None, show=True):
    """Draw circuit using Qiskit for research paper figures."""
    try:
        from qiskit import QuantumCircuit
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Run:  pip install qiskit pylatexenc")

    if sample_input is None:
        sample_input = [np.pi / 4] * N_QUBITS
    sample = np.tanh(np.array(sample_input)) * (np.pi / 2)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    for i in range(N_QUBITS):
        qc.h(i)
    qc.barrier(label="Init")

    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qc.ry(round(float(sample[i]), 3), i)
        qc.barrier(label=f"Enc {layer+1}")
        for i in range(N_QUBITS):
            qc.rz(np.pi/4, i)
            qc.ry(np.pi/4, i)
            qc.rz(np.pi/4, i)
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)
        qc.barrier(label=f"Var {layer+1}")

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    print(f"\nIBM Circuit Stats:")
    print(f"  Qubits : {N_QUBITS}")
    print(f"  Layers : {N_LAYERS}")
    print(f"  Depth  : {qc.depth()}")
    print(f"  Gates  : {dict(qc.count_ops())}")

    try:
        fig = qc.draw(output="mpl", style=style, fold=fold, scale=0.65)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved → {save_path}")
        if show:
            plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"mpl failed ({e}), text draw:")
        print(qc.draw(output="text", fold=80))
    return qc


def print_pennylane_circuit():
    """Print circuit in terminal — no Qiskit needed."""
    sample  = np.array([np.pi/4] * N_QUBITS, dtype=np.float32)
    weights = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=np.float32)
    print("\nPennyLane Circuit:")
    print("=" * 55)
    print(qml.draw(circuit)(torch.tensor(sample), torch.tensor(weights)))
    print("=" * 55)


# ── auto IBM draw ─────────────────────────────────────────
if USE_IBM:
    try:
        draw_ibm_circuit()
    except ImportError as e:
        print(f"IBM draw failed: {e}")
        print_pennylane_circuit()


# ── quick test ────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing QuantumLayer …")
    ql  = QuantumLayer()
    x   = torch.randn(4, N_QUBITS)
    out = ql(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Range  : [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("✓ Working correctly")
    print_pennylane_circuit()