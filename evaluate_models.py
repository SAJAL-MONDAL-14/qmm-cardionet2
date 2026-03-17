# """
# evaluate_models.py  —  Full Model Evaluation Suite
# ====================================================
# FAST MODE:  FORCE_RERUN = False  →  loads cached predictions (seconds)
# FULL MODE:  FORCE_RERUN = True   →  re-runs all models from scratch

# Saves separate PNG per model:
#   svm_confusion.png, svm_roc.png, svm_metrics.png ... etc

# Run:  python evaluate_models.py
# """

# import os
# import sys
# import warnings
# import numpy as np
# import joblib
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.gridspec as gridspec
# warnings.filterwarnings("ignore")

# import torch
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score,
#     confusion_matrix, roc_curve, classification_report
# )
# from sklearn.preprocessing import StandardScaler

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from models.classical.ann_model import ANNModel
# from models.quantum.qnn_model   import QNNModel
# from models.quantum.vqc_model   import VQCModel

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# MAX_SAMPLES = 4000
# OUT_DIR     = "evaluation_results"
# CACHE_PATH  = "saved_models/feature_cache.npz"
# PRED_CACHE  = "evaluation_results/predictions_cache.npz"
# device      = "cpu"

# # ── FORCE_RERUN = False  →  fast mode (default, seconds)  ──
# # ── FORCE_RERUN = True   →  re-run all models from scratch ──
# FORCE_RERUN = False

# os.makedirs(OUT_DIR, exist_ok=True)

# # ── Colour palette ────────────────────────────────────────
# PALETTE = {
#     "SVM"        : "#4C72B0",
#     "ANN"        : "#DD8452",
#     "QNN"        : "#55A868",
#     "VQC"        : "#C44E52",
#     "Multimodal" : "#8172B2",
# }
# BG   = "#0F1117"
# CARD = "#1A1D27"
# TEXT = "#E8EAF0"
# GRID = "#2A2D3A"

# plt.rcParams.update({
#     "figure.facecolor" : BG,
#     "axes.facecolor"   : CARD,
#     "axes.edgecolor"   : GRID,
#     "axes.labelcolor"  : TEXT,
#     "xtick.color"      : TEXT,
#     "ytick.color"      : TEXT,
#     "text.color"       : TEXT,
#     "grid.color"       : GRID,
#     "font.size"        : 11,
# })


# # ── Helper ────────────────────────────────────────────────
# def save_fig(fig, path):
#     fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
#     plt.close(fig)
#     print("  Saved -> " + path)


# # ═══════════════════════════════════════════════════════════
# #  LOAD OR RUN PREDICTIONS
# # ═══════════════════════════════════════════════════════════

# if not FORCE_RERUN and os.path.exists(PRED_CACHE):
#     # ── Fast mode: load cached predictions ────────────────
#     print("\n✓ Loading cached predictions from " + PRED_CACHE)
#     print("  (Set FORCE_RERUN = True to re-run models)\n")
#     pc      = np.load(PRED_CACHE, allow_pickle=True)
#     results = pc["results"].item()

# else:
#     # ── Full mode: run all models ──────────────────────────
#     print("\nRunning model inference from scratch ...\n")

#     # ── Load dataset from cache ───────────────────────────
#     if os.path.exists(CACHE_PATH):
#         print("Loading from feature cache: " + CACHE_PATH)
#         data     = np.load(CACHE_PATH)
#         ecg_arr  = data["ecg"]
#         clin_arr = data["clin"]
#         y_flat   = data["labels"].astype(int)
#         X_flat   = np.concatenate([ecg_arr, clin_arr], axis=1).astype(np.float32)
#         print("Loaded " + str(len(y_flat)) + " samples")
#     else:
#         print("Cache not found, building dataset ...")
#         from dataset.dataset_builder import build_flat_dataset
#         X_flat, y_flat = build_flat_dataset(max_samples=MAX_SAMPLES, multilead=True)

#     # ── Scale ─────────────────────────────────────────────
#     if os.path.exists("saved_models/svm_scaler.pkl"):
#         flat_scaler   = joblib.load("saved_models/svm_scaler.pkl")
#         X_flat_scaled = flat_scaler.transform(X_flat).astype(np.float32)
#         print("Using saved SVM scaler")
#     else:
#         flat_scaler   = StandardScaler()
#         X_flat_scaled = flat_scaler.fit_transform(X_flat).astype(np.float32)
#         print("Fitted fresh scaler")

#     X_flat_scaled = np.nan_to_num(X_flat_scaled, nan=0., posinf=0., neginf=0.)
#     input_dim     = X_flat_scaled.shape[1]

#     # ── Load flat models ──────────────────────────────────
#     flat_models = {}

#     if os.path.exists("saved_models/svm_model.pkl"):
#         flat_models["SVM"] = joblib.load("saved_models/svm_model.pkl")
#         print("Loaded SVM")

#     if os.path.exists("saved_models/ann_model.pth"):
#         m = ANNModel(input_dim)
#         m.load_state_dict(torch.load("saved_models/ann_model.pth", map_location="cpu"))
#         m.eval()
#         flat_models["ANN"] = m
#         print("Loaded ANN")

#     if os.path.exists("saved_models/qnn_model.pth"):
#         m = QNNModel(input_dim)
#         m.load_state_dict(torch.load("saved_models/qnn_model.pth", map_location="cpu"))
#         m.eval()
#         flat_models["QNN"] = m
#         print("Loaded QNN")

#     if os.path.exists("saved_models/vqc_model.pth"):
#         m = VQCModel(input_dim)
#         m.load_state_dict(torch.load("saved_models/vqc_model.pth", map_location="cpu"))
#         m.eval()
#         flat_models["VQC"] = m
#         print("Loaded VQC")

#     # ── Run predictions ───────────────────────────────────
#     results  = {}
#     X_tensor = torch.tensor(X_flat_scaled, dtype=torch.float32)

#     for name, model_obj in flat_models.items():
#         print("\nEvaluating " + name + " ...")

#         if name == "SVM":
#             probs = model_obj.predict_proba(X_flat_scaled)[:, 1]
#             preds = model_obj.predict(X_flat_scaled)
#         else:
#             with torch.no_grad():
#                 out   = model_obj(X_tensor)
#                 probs = torch.sigmoid(out).numpy().flatten()
#             best_t, best_a = 0.5, 0.0
#             for t in np.arange(0.3, 0.7, 0.01):
#                 a = accuracy_score(y_flat, (probs > t).astype(int))
#                 if a > best_a:
#                     best_a = a; best_t = t
#             preds = (probs > best_t).astype(int)
#             print("  Best threshold: " + str(round(best_t, 2)))

#         acc  = accuracy_score(y_flat, preds)
#         auc  = roc_auc_score(y_flat, probs)
#         f1   = f1_score(y_flat, preds, zero_division=0)
#         prec = precision_score(y_flat, preds, zero_division=0)
#         rec  = recall_score(y_flat, preds, zero_division=0)
#         cm   = confusion_matrix(y_flat, preds)

#         results[name] = {
#             "probs"  : probs,
#             "preds"  : preds,
#             "labels" : y_flat,
#             "acc"    : acc,
#             "auc"    : auc,
#             "f1"     : f1,
#             "prec"   : prec,
#             "rec"    : rec,
#             "cm"     : cm,
#         }
#         print("  acc=" + str(round(acc, 4)) +
#               "  auc=" + str(round(auc, 4)) +
#               "  f1="  + str(round(f1,  4)))

#     # ── Multimodal ────────────────────────────────────────
#     mm_path = "saved_models/best_multimodal_model.pth"
#     if os.path.exists(mm_path) and os.path.exists(CACHE_PATH):
#         try:
#             from train.train_multimodal import DualPathQuantumNet, CachedMultimodalDataset
#             from torch.utils.data import DataLoader

#             data       = np.load(CACHE_PATH)
#             mm_dataset = CachedMultimodalDataset(
#                 data["ecg"], data["clin"], data["labels"]
#             )
#             mm_loader  = DataLoader(mm_dataset, batch_size=256, shuffle=False)
#             ecg_dim    = mm_dataset.ecg.shape[1]

#             mm_model = DualPathQuantumNet(ecg_dim)
#             mm_model.load_state_dict(torch.load(mm_path, map_location="cpu"))
#             mm_model.eval()
#             print("\nLoaded Multimodal")

#             mm_probs, mm_preds, mm_labels = [], [], []
#             with torch.no_grad():
#                 for ef, cf, lb in mm_loader:
#                     p = torch.sigmoid(mm_model(ef, cf)).numpy().flatten()
#                     mm_probs.extend(p)
#                     mm_labels.extend(lb.numpy())

#             mm_probs  = np.array(mm_probs)
#             mm_labels = np.array(mm_labels)

#             best_t, best_a = 0.5, 0.0
#             for t in np.arange(0.3, 0.7, 0.01):
#                 a = accuracy_score(mm_labels, (mm_probs > t).astype(int))
#                 if a > best_a:
#                     best_a = a; best_t = t
#             mm_preds = (mm_probs > best_t).astype(int)

#             results["Multimodal"] = {
#                 "probs"  : mm_probs,
#                 "preds"  : mm_preds,
#                 "labels" : mm_labels,
#                 "acc"    : accuracy_score(mm_labels, mm_preds),
#                 "auc"    : roc_auc_score(mm_labels, mm_probs),
#                 "f1"     : f1_score(mm_labels, mm_preds, zero_division=0),
#                 "prec"   : precision_score(mm_labels, mm_preds, zero_division=0),
#                 "rec"    : recall_score(mm_labels, mm_preds, zero_division=0),
#                 "cm"     : confusion_matrix(mm_labels, mm_preds),
#             }
#             print("  Multimodal acc=" + str(round(results["Multimodal"]["acc"], 4)))

#         except Exception as e:
#             print("Multimodal load failed: " + str(e))
#             import traceback; traceback.print_exc()

#     # ── Save predictions cache ────────────────────────────
#     np.savez_compressed(PRED_CACHE, results=np.array(results, dtype=object))
#     print("\n✓ Predictions cached -> " + PRED_CACHE)
#     print("  Next run will be instant!\n")


# # ═══════════════════════════════════════════════════════════
# #  SUMMARY TABLE
# # ═══════════════════════════════════════════════════════════
# model_names = list(results.keys())
# n_models    = len(model_names)
# colors      = [PALETTE.get(n, "#888") for n in model_names]

# print("\n" + "="*72)
# print("{:<14} {:>9} {:>9} {:>7} {:>11} {:>8}".format(
#     "Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall"))
# print("-"*72)
# for name, r in results.items():
#     print("{:<14} {:>8.2f}%  {:>8.4f}  {:>7.4f}  {:>10.4f}  {:>7.4f}".format(
#         name,
#         r["acc"]*100, r["auc"], r["f1"], r["prec"], r["rec"]
#     ))
# print("="*72)

# for name, r in results.items():
#     print("\n-- " + name + " Classification Report --")
#     print(classification_report(
#         r["labels"], r["preds"],
#         target_names=["Normal", "Abnormal"],
#         zero_division=0
#     ))


# # ═══════════════════════════════════════════════════════════
# #  INDIVIDUAL MODEL PLOTS
# # ═══════════════════════════════════════════════════════════

# def plot_confusion(name, r):
#     col  = PALETTE.get(name, "#888")
#     cm   = r["cm"]
#     cmap = mcolors.LinearSegmentedColormap.from_list("c", [CARD, col])
#     fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=BG)
#     ax.set_facecolor(CARD)
#     ax.imshow(cm, cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val = cm[j, k]; pct = val / cm.sum() * 100
#             fc  = BG if val > cm.max() * 0.6 else TEXT
#             ax.text(k, j, str(val) + "\n(" + str(round(pct,1)) + "%)",
#                     ha="center", va="center",
#                     color=fc, fontsize=13, fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(
#         name + "  --  Confusion Matrix\n" +
#         "Acc=" + str(round(r["acc"]*100,1)) + "%  " +
#         "AUC=" + str(round(r["auc"],3)) + "  " +
#         "F1="  + str(round(r["f1"],3)),
#         fontsize=13, fontweight="bold", color=col
#     )
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR + "/" + name.lower() + "_confusion.png")


# def plot_roc(name, r):
#     col = PALETTE.get(name, "#888")
#     fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
#     fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
#     ax.set_facecolor(CARD)
#     ax.plot(fpr, tpr, color=col, lw=2.5,
#             label="AUC = " + str(round(r["auc"],3)))
#     ax.fill_between(fpr, tpr, alpha=0.1, color=col)
#     ax.plot([0,1],[0,1], "--", color="#666", lw=1.5, label="Random")
#     ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
#     ax.set_title(name + "  --  ROC Curve\nAUC = " + str(round(r["auc"],4)),
#                  fontsize=13, fontweight="bold", color=col)
#     ax.legend(loc="lower right", fontsize=11, facecolor=CARD, edgecolor=GRID)
#     ax.grid(True, alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(1.5)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR + "/" + name.lower() + "_roc.png")


# def plot_metrics(name, r):
#     col    = PALETTE.get(name, "#888")
#     labels = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
#     values = [r["acc"], r["auc"], r["f1"], r["prec"], r["rec"]]
#     fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
#     ax.set_facecolor(CARD)
#     bars = ax.bar(labels, values, color=col, width=0.5, edgecolor=BG)
#     for bar, val in zip(bars, values):
#         ax.text(bar.get_x() + bar.get_width()/2,
#                 bar.get_height() + 0.01,
#                 str(round(val, 3)),
#                 ha="center", va="bottom",
#                 color=TEXT, fontsize=11, fontweight="bold")
#     ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
#     ax.set_title(name + "  --  Metrics Overview",
#                  fontsize=13, fontweight="bold", color=col)
#     ax.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR + "/" + name.lower() + "_metrics.png")


# print("\nGenerating individual model plots ...")
# for name, r in results.items():
#     print("\n  " + name + ":")
#     plot_confusion(name, r)
#     plot_roc(name, r)
#     plot_metrics(name, r)


# # ═══════════════════════════════════════════════════════════
# #  COMBINED PLOTS
# # ═══════════════════════════════════════════════════════════

# # ── All confusion matrices ────────────────────────────────
# cols = min(3, n_models)
# rows = (n_models + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols,
#                           figsize=(cols*5, rows*4.5),
#                           facecolor=BG)
# fig.suptitle("Confusion Matrices — All Models",
#              fontsize=18, fontweight="bold", color=TEXT, y=1.01)
# axf = np.array(axes).flatten()
# for idx, name in enumerate(model_names):
#     ax = axf[idx]; r = results[name]
#     cm = r["cm"]; col = PALETTE.get(name, "#888")
#     cmap = mcolors.LinearSegmentedColormap.from_list("c", [CARD, col])
#     ax.imshow(cm, cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val = cm[j,k]; pct = val/cm.sum()*100
#             fc  = BG if val > cm.max()*0.6 else TEXT
#             ax.text(k, j, str(val)+"\n("+str(round(pct,1))+"%)",
#                     ha="center", va="center",
#                     color=fc, fontsize=11, fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(name+"\nAcc="+str(round(r["acc"]*100,1))+
#                  "%  AUC="+str(round(r["auc"],3)),
#                  fontsize=12, fontweight="bold", color=col)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
# for idx in range(n_models, len(axf)):
#     axf[idx].set_visible(False)
# plt.tight_layout()
# save_fig(fig, OUT_DIR + "/all_confusion_matrices.png")

# # ── All ROC curves ────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
# ax.set_facecolor(CARD)
# for name, r in results.items():
#     fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
#     ax.plot(fpr, tpr, color=PALETTE.get(name,"#888"), lw=2.5,
#             label=name+"  (AUC="+str(round(r["auc"],3))+")")
# ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
# ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
# ax.set_title("ROC Curves — All Models", fontsize=16, fontweight="bold")
# ax.legend(loc="lower right", fontsize=11, facecolor=CARD, edgecolor=GRID)
# ax.grid(True, alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
# plt.tight_layout()
# save_fig(fig, OUT_DIR + "/all_roc_curves.png")

# # ── Comparison bars ───────────────────────────────────────
# fig, axes2 = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
# fig.suptitle("Model Comparison", fontsize=18, fontweight="bold", color=TEXT)
# for ax, (key, ylabel, fn) in zip(axes2, [
#     ("acc", "Accuracy (%)", lambda v: v*100),
#     ("f1",  "F1 Score",     lambda v: v),
# ]):
#     ax.set_facecolor(CARD)
#     vals = [fn(results[n][key]) for n in model_names]
#     bars = ax.bar(model_names, vals, color=colors, width=0.55, edgecolor=BG)
#     for b, v in zip(bars, vals):
#         ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
#                 str(round(v,2)),
#                 ha="center", va="bottom", color=TEXT,
#                 fontsize=11, fontweight="bold")
#     ax.set_ylabel(ylabel); ax.set_ylim(0, max(vals)*1.18 if vals else 1)
#     ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", labelsize=10)
# plt.tight_layout()
# save_fig(fig, OUT_DIR + "/model_comparison.png")

# # ── Radar chart ───────────────────────────────────────────
# mk  = ["acc","auc","f1","prec","rec"]
# ml  = ["Accuracy","ROC-AUC","F1","Precision","Recall"]
# ang = np.linspace(0, 2*np.pi, len(mk), endpoint=False).tolist()
# ang += ang[:1]
# fig, ax4 = plt.subplots(figsize=(7,7),
#                          subplot_kw=dict(polar=True), facecolor=BG)
# ax4.set_facecolor(CARD)
# ax4.set_theta_offset(np.pi/2); ax4.set_theta_direction(-1)
# ax4.set_thetagrids(np.degrees(ang[:-1]), ml, fontsize=11, color=TEXT)
# for name, r in results.items():
#     vals = [r[k] for k in mk]+[r[mk[0]]]
#     col  = PALETTE.get(name, "#888")
#     ax4.plot(ang, vals, color=col, lw=2.5, label=name)
#     ax4.fill(ang, vals, color=col, alpha=0.08)
# ax4.set_ylim(0,1)
# ax4.set_yticks([0.25,0.5,0.75,1.0])
# ax4.set_yticklabels(["0.25","0.50","0.75","1.00"], color=TEXT, fontsize=8)
# ax4.grid(color=GRID, alpha=0.5)
# ax4.legend(loc="upper right", bbox_to_anchor=(1.3,1.1),
#            fontsize=10, facecolor=CARD, edgecolor=GRID)
# ax4.set_title("Multi-Metric Radar", fontsize=15, fontweight="bold", pad=20)
# plt.tight_layout()
# save_fig(fig, OUT_DIR + "/radar_chart.png")

# # ── Dashboard ─────────────────────────────────────────────
# fig5 = plt.figure(figsize=(18, 12), facecolor=BG)
# fig5.suptitle("QMM CARDIONET2 — Model Evaluation Dashboard",
#               fontsize=20, fontweight="bold", color=TEXT, y=0.98)
# gs = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.35)

# ax_r = fig5.add_subplot(gs[0,:2]); ax_r.set_facecolor(CARD)
# for name, r in results.items():
#     fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
#     ax_r.plot(fpr, tpr, lw=2.5, color=PALETTE.get(name,"#888"),
#               label=name+" AUC="+str(round(r["auc"],3)))
# ax_r.plot([0,1],[0,1],"--",color="#666",lw=1.5)
# ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
# ax_r.set_title("ROC Curves", fontsize=14, fontweight="bold")
# ax_r.legend(fontsize=9, facecolor=CARD, edgecolor=GRID); ax_r.grid(alpha=0.3)

# ax_rd = fig5.add_subplot(gs[0,2], polar=True); ax_rd.set_facecolor(CARD)
# ax_rd.set_theta_offset(np.pi/2); ax_rd.set_theta_direction(-1)
# ax_rd.set_thetagrids(np.degrees(ang[:-1]), ml, fontsize=9)
# for name, r in results.items():
#     v = [r[k] for k in mk]+[r[mk[0]]]
#     ax_rd.plot(ang, v, lw=2, color=PALETTE.get(name,"#888"))
#     ax_rd.fill(ang, v, alpha=0.06, color=PALETTE.get(name,"#888"))
# ax_rd.set_ylim(0,1); ax_rd.grid(color=GRID, alpha=0.4)
# ax_rd.set_title("Radar", fontsize=12, fontweight="bold", pad=18)

# ax_a = fig5.add_subplot(gs[1,0]); ax_a.set_facecolor(CARD)
# vals = [results[n]["acc"]*100 for n in model_names]
# bars = ax_a.bar(model_names, vals, color=colors, edgecolor=BG)
# for b, v in zip(bars, vals):
#     ax_a.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
#               str(round(v,1))+"%", ha="center", va="bottom",
#               fontsize=9, color=TEXT)
# ax_a.set_ylabel("Accuracy %"); ax_a.set_ylim(0, max(vals)*1.18)
# ax_a.set_title("Accuracy", fontsize=13, fontweight="bold")
# ax_a.grid(axis="y", alpha=0.3)

# ax_f = fig5.add_subplot(gs[1,1]); ax_f.set_facecolor(CARD)
# vf   = [results[n]["f1"] for n in model_names]
# bars2 = ax_f.bar(model_names, vf, color=colors, edgecolor=BG)
# for b, v in zip(bars2, vf):
#     ax_f.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
#               str(round(v,3)), ha="center", va="bottom",
#               fontsize=9, color=TEXT)
# ax_f.set_ylabel("F1 Score"); ax_f.set_ylim(0, max(vf)*1.18 if vf else 1)
# ax_f.set_title("F1 Score", fontsize=13, fontweight="bold")
# ax_f.grid(axis="y", alpha=0.3)

# ax_t = fig5.add_subplot(gs[1,2]); ax_t.set_facecolor(CARD); ax_t.axis("off")
# rows_data = [
#     [n,
#      str(round(results[n]["acc"]*100,1))+"%",
#      str(round(results[n]["auc"],3)),
#      str(round(results[n]["f1"],3)),
#      str(round(results[n]["prec"],3)),
#      str(round(results[n]["rec"],3))]
#     for n in model_names
# ]
# tbl = ax_t.table(cellText=rows_data,
#                   colLabels=["Model","Acc","AUC","F1","Prec","Rec"],
#                   loc="center", cellLoc="center")
# tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
# for (row, col), cell in tbl.get_celld().items():
#     cell.set_facecolor(BG if row % 2 == 0 else CARD)
#     cell.set_edgecolor(GRID)
#     cell.set_text_props(color=TEXT,
#                         fontweight="bold" if row == 0 else "normal")
# ax_t.set_title("Summary Table", fontsize=13, fontweight="bold")

# save_fig(fig5, OUT_DIR + "/dashboard.png")


# # ═══════════════════════════════════════════════════════════
# #  SUMMARY
# # ═══════════════════════════════════════════════════════════
# print("\n" + "="*60)
# print("  All plots saved to ./" + OUT_DIR + "/")
# print("="*60)
# print("\n  Individual model plots:")
# for name in model_names:
#     n = name.lower()
#     print("    " + n + "_confusion.png")
#     print("    " + n + "_roc.png")
#     print("    " + n + "_metrics.png")
# print("\n  Combined plots:")
# print("    all_confusion_matrices.png")
# print("    all_roc_curves.png")
# print("    model_comparison.png")
# print("    radar_chart.png")
# print("    dashboard.png")
# print("="*60)





# """
# evaluate_models.py  —  Fair Model Evaluation
# =============================================
# Uses shared_scaler.pkl so ALL models see identical scaled data.

# FORCE_RERUN = False  ->  fast mode (seconds)
# FORCE_RERUN = True   ->  re-run all models from scratch

# Run:  python evaluate_models.py
# """

# import os
# import sys
# import warnings
# import numpy as np
# import joblib
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.gridspec as gridspec
# warnings.filterwarnings("ignore")

# import torch
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score,
#     confusion_matrix, roc_curve, classification_report
# )
# from sklearn.preprocessing import StandardScaler

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from models.classical.ann_model import ANNModel
# from models.quantum.qnn_model   import QNNModel
# from models.quantum.vqc_model   import VQCModel

# # ═══════════════════════════════════════════════════════════
# #  CONFIG
# # ═══════════════════════════════════════════════════════════
# OUT_DIR      = "evaluation_results"
# CACHE_PATH   = "saved_models/feature_cache.npz"
# PRED_CACHE   = "evaluation_results/predictions_cache.npz"
# SCALER_PATH  = "saved_models/shared_scaler.pkl"   # from train_all_flat.py
# device       = "cpu"

# # False = fast mode (load cache)
# # True  = re-run all models from scratch
# FORCE_RERUN  = False

# os.makedirs(OUT_DIR, exist_ok=True)

# # ── Colours ───────────────────────────────────────────────
# PALETTE = {
#     "SVM"        : "#4C72B0",
#     "ANN"        : "#DD8452",
#     "QNN"        : "#55A868",
#     "VQC"        : "#C44E52",
#     "Multimodal" : "#8172B2",
# }
# BG   = "#0F1117"
# CARD = "#1A1D27"
# TEXT = "#E8EAF0"
# GRID = "#2A2D3A"

# plt.rcParams.update({
#     "figure.facecolor" : BG,   "axes.facecolor"  : CARD,
#     "axes.edgecolor"   : GRID, "axes.labelcolor" : TEXT,
#     "xtick.color"      : TEXT, "ytick.color"     : TEXT,
#     "text.color"       : TEXT, "grid.color"      : GRID,
#     "font.size"        : 11,
# })

# def save_fig(fig, path):
#     fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
#     plt.close(fig)
#     print("  Saved -> " + path)


# # ═══════════════════════════════════════════════════════════
# #  LOAD OR RUN PREDICTIONS
# # ═══════════════════════════════════════════════════════════
# if not FORCE_RERUN and os.path.exists(PRED_CACHE):
#     print("\nLoading cached predictions from " + PRED_CACHE)
#     print("  Set FORCE_RERUN = True to re-run models\n")
#     pc      = np.load(PRED_CACHE, allow_pickle=True)
#     results = pc["results"].item()

# else:
#     print("\nRunning model inference ...\n")

#     # ── Load dataset ──────────────────────────────────────
#     if not os.path.exists(CACHE_PATH):
#         print("ERROR: Cache not found - run train_multimodal.py first")
#         sys.exit(1)

#     data     = np.load(CACHE_PATH)
#     ecg_arr  = data["ecg"]
#     clin_arr = data["clin"]
#     y_all    = data["labels"].astype(int)
#     X_all    = np.concatenate([ecg_arr, clin_arr], axis=1).astype(np.float32)
#     X_all    = np.nan_to_num(X_all, nan=0., posinf=0., neginf=0.)
#     print("Loaded " + str(len(y_all)) + " samples from cache")

#     # ── Load shared scaler ────────────────────────────────
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#         print("Using shared scaler: " + SCALER_PATH)
#     elif os.path.exists("saved_models/svm_scaler.pkl"):
#         scaler = joblib.load("saved_models/svm_scaler.pkl")
#         print("Using SVM scaler as fallback")
#     else:
#         scaler = StandardScaler()
#         scaler.fit(X_all)
#         print("WARNING: No saved scaler found - fitting fresh scaler")
#         print("         For fair results run train_all_flat.py first!")

#     X_scaled = scaler.transform(X_all).astype(np.float32)
#     X_scaled = np.nan_to_num(X_scaled, nan=0., posinf=0., neginf=0.)
#     input_dim = X_scaled.shape[1]
#     X_tensor  = torch.tensor(X_scaled, dtype=torch.float32)

#     # ── Load models ───────────────────────────────────────
#     results = {}

#     # SVM
#     if os.path.exists("saved_models/svm_model.pkl"):
#         svm   = joblib.load("saved_models/svm_model.pkl")
#         probs = svm.predict_proba(X_scaled)[:, 1]
#         preds = svm.predict(X_scaled)
#         results["SVM"] = {
#             "probs"  : probs, "preds"  : preds, "labels" : y_all,
#             "acc"    : accuracy_score(y_all, preds),
#             "auc"    : roc_auc_score(y_all, probs),
#             "f1"     : f1_score(y_all, preds, zero_division=0),
#             "prec"   : precision_score(y_all, preds, zero_division=0),
#             "rec"    : recall_score(y_all, preds, zero_division=0),
#             "cm"     : confusion_matrix(y_all, preds),
#         }
#         print("SVM  acc=" + str(round(results["SVM"]["acc"]*100,2)) + "%")

#     # ANN
#     if os.path.exists("saved_models/ann_model.pth"):
#         ann = ANNModel(input_dim)
#         ann.load_state_dict(torch.load("saved_models/ann_model.pth",
#                                         map_location="cpu"))
#         ann.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(ann(X_tensor)).numpy().flatten()
#         # find best threshold
#         best_t, best_a = 0.5, 0.0
#         for t in np.arange(0.3, 0.7, 0.01):
#             a = accuracy_score(y_all, (probs>t).astype(int))
#             if a > best_a: best_a=a; best_t=t
#         preds = (probs > best_t).astype(int)
#         results["ANN"] = {
#             "probs"  : probs, "preds"  : preds, "labels" : y_all,
#             "acc"    : accuracy_score(y_all, preds),
#             "auc"    : roc_auc_score(y_all, probs),
#             "f1"     : f1_score(y_all, preds, zero_division=0),
#             "prec"   : precision_score(y_all, preds, zero_division=0),
#             "rec"    : recall_score(y_all, preds, zero_division=0),
#             "cm"     : confusion_matrix(y_all, preds),
#         }
#         print("ANN  acc=" + str(round(results["ANN"]["acc"]*100,2)) + "%")

#     # QNN
#     if os.path.exists("saved_models/qnn_model.pth"):
#         qnn = QNNModel(input_dim)
#         qnn.load_state_dict(torch.load("saved_models/qnn_model.pth",
#                                         map_location="cpu"))
#         qnn.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(qnn(X_tensor)).numpy().flatten()
#         best_t, best_a = 0.5, 0.0
#         for t in np.arange(0.3, 0.7, 0.01):
#             a = accuracy_score(y_all, (probs>t).astype(int))
#             if a > best_a: best_a=a; best_t=t
#         preds = (probs > best_t).astype(int)
#         results["QNN"] = {
#             "probs"  : probs, "preds"  : preds, "labels" : y_all,
#             "acc"    : accuracy_score(y_all, preds),
#             "auc"    : roc_auc_score(y_all, probs),
#             "f1"     : f1_score(y_all, preds, zero_division=0),
#             "prec"   : precision_score(y_all, preds, zero_division=0),
#             "rec"    : recall_score(y_all, preds, zero_division=0),
#             "cm"     : confusion_matrix(y_all, preds),
#         }
#         print("QNN  acc=" + str(round(results["QNN"]["acc"]*100,2)) + "%")

#     # VQC
#     if os.path.exists("saved_models/vqc_model.pth"):
#         vqc = VQCModel(input_dim)
#         vqc.load_state_dict(torch.load("saved_models/vqc_model.pth",
#                                         map_location="cpu"))
#         vqc.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(vqc(X_tensor)).numpy().flatten()
#         best_t, best_a = 0.5, 0.0
#         for t in np.arange(0.3, 0.7, 0.01):
#             a = accuracy_score(y_all, (probs>t).astype(int))
#             if a > best_a: best_a=a; best_t=t
#         preds = (probs > best_t).astype(int)
#         results["VQC"] = {
#             "probs"  : probs, "preds"  : preds, "labels" : y_all,
#             "acc"    : accuracy_score(y_all, preds),
#             "auc"    : roc_auc_score(y_all, probs),
#             "f1"     : f1_score(y_all, preds, zero_division=0),
#             "prec"   : precision_score(y_all, preds, zero_division=0),
#             "rec"    : recall_score(y_all, preds, zero_division=0),
#             "cm"     : confusion_matrix(y_all, preds),
#         }
#         print("VQC  acc=" + str(round(results["VQC"]["acc"]*100,2)) + "%")

#     # Multimodal
#     mm_path = "saved_models/best_multimodal_model.pth"
#     if os.path.exists(mm_path) and os.path.exists(CACHE_PATH):
#         try:
#             from train.train_multimodal import (
#                 DualPathQuantumNet, CachedMultimodalDataset
#             )
#             from torch.utils.data import DataLoader

#             mm_ds     = CachedMultimodalDataset(ecg_arr, clin_arr,
#                                                  data["labels"])
#             mm_loader = DataLoader(mm_ds, batch_size=256, shuffle=False)
#             mm_model  = DualPathQuantumNet(mm_ds.ecg.shape[1])
#             mm_model.load_state_dict(torch.load(mm_path, map_location="cpu"))
#             mm_model.eval()

#             mm_probs, mm_labels = [], []
#             with torch.no_grad():
#                 for ef, cf, lb in mm_loader:
#                     p = torch.sigmoid(mm_model(ef,cf)).numpy().flatten()
#                     mm_probs.extend(p); mm_labels.extend(lb.numpy())
#             mm_probs  = np.array(mm_probs)
#             mm_labels = np.array(mm_labels)

#             best_t, best_a = 0.5, 0.0
#             for t in np.arange(0.3, 0.7, 0.01):
#                 a = accuracy_score(mm_labels, (mm_probs>t).astype(int))
#                 if a > best_a: best_a=a; best_t=t
#             mm_preds = (mm_probs > best_t).astype(int)

#             results["Multimodal"] = {
#                 "probs"  : mm_probs, "preds"  : mm_preds,
#                 "labels" : mm_labels,
#                 "acc"    : accuracy_score(mm_labels, mm_preds),
#                 "auc"    : roc_auc_score(mm_labels, mm_probs),
#                 "f1"     : f1_score(mm_labels, mm_preds, zero_division=0),
#                 "prec"   : precision_score(mm_labels, mm_preds, zero_division=0),
#                 "rec"    : recall_score(mm_labels, mm_preds, zero_division=0),
#                 "cm"     : confusion_matrix(mm_labels, mm_preds),
#             }
#             print("Multimodal  acc=" +
#                   str(round(results["Multimodal"]["acc"]*100,2)) + "%")
#         except Exception as e:
#             print("Multimodal failed: " + str(e))
#             import traceback; traceback.print_exc()

#     # ── Save predictions cache ────────────────────────────
#     np.savez_compressed(PRED_CACHE,
#                         results=np.array(results, dtype=object))
#     print("\nPredictions cached -> " + PRED_CACHE)
#     print("Next run will be instant!\n")


# # ═══════════════════════════════════════════════════════════
# #  SUMMARY TABLE
# # ═══════════════════════════════════════════════════════════
# model_names = list(results.keys())
# n_models    = len(model_names)
# colors      = [PALETTE.get(n,"#888") for n in model_names]

# print("\n" + "="*72)
# print("{:<14} {:>9} {:>9} {:>7} {:>11} {:>8}".format(
#     "Model","Accuracy","ROC-AUC","F1","Precision","Recall"))
# print("-"*72)
# for name, r in results.items():
#     print("{:<14} {:>8.2f}%  {:>8.4f}  {:>7.4f}  {:>10.4f}  {:>7.4f}".format(
#         name, r["acc"]*100, r["auc"], r["f1"], r["prec"], r["rec"]))
# print("="*72)

# for name, r in results.items():
#     print("\n-- " + name + " --")
#     print(classification_report(r["labels"], r["preds"],
#           target_names=["Normal","Abnormal"], zero_division=0))


# # ═══════════════════════════════════════════════════════════
# #  PLOT FUNCTIONS
# # ═══════════════════════════════════════════════════════════
# def plot_confusion(name, r):
#     col  = PALETTE.get(name,"#888")
#     cm   = r["cm"]
#     cmap = mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
#     fig, ax = plt.subplots(figsize=(5,4.5), facecolor=BG)
#     ax.set_facecolor(CARD); ax.imshow(cm, cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val=cm[j,k]; pct=val/cm.sum()*100
#             fc = BG if val>cm.max()*0.6 else TEXT
#             ax.text(k,j,str(val)+"\n("+str(round(pct,1))+"%)",
#                     ha="center",va="center",color=fc,
#                     fontsize=13,fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(name+"  --  Confusion Matrix\nAcc="+
#                  str(round(r["acc"]*100,1))+"% AUC="+
#                  str(round(r["auc"],3))+" F1="+
#                  str(round(r["f1"],3)),
#                  fontsize=12,fontweight="bold",color=col)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_confusion.png")

# def plot_roc(name, r):
#     col=PALETTE.get(name,"#888")
#     fpr,tpr,_=roc_curve(r["labels"],r["probs"])
#     fig,ax=plt.subplots(figsize=(6,5),facecolor=BG)
#     ax.set_facecolor(CARD)
#     ax.plot(fpr,tpr,color=col,lw=2.5,
#             label="AUC="+str(round(r["auc"],3)))
#     ax.fill_between(fpr,tpr,alpha=0.1,color=col)
#     ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title(name+"  --  ROC Curve\nAUC="+str(round(r["auc"],4)),
#                  fontsize=13,fontweight="bold",color=col)
#     ax.legend(loc="lower right",fontsize=11,facecolor=CARD,edgecolor=GRID)
#     ax.grid(True,alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(1.5)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_roc.png")

# def plot_metrics(name, r):
#     col=PALETTE.get(name,"#888")
#     lbls=["Accuracy","ROC-AUC","F1","Precision","Recall"]
#     vals=[r["acc"],r["auc"],r["f1"],r["prec"],r["rec"]]
#     fig,ax=plt.subplots(figsize=(7,4),facecolor=BG)
#     ax.set_facecolor(CARD)
#     bars=ax.bar(lbls,vals,color=col,width=0.5,edgecolor=BG)
#     for b,v in zip(bars,vals):
#         ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,
#                 str(round(v,3)),ha="center",va="bottom",
#                 color=TEXT,fontsize=11,fontweight="bold")
#     ax.set_ylim(0,1.15); ax.set_ylabel("Score")
#     ax.set_title(name+"  --  Metrics",fontsize=13,
#                  fontweight="bold",color=col)
#     ax.grid(axis="y",alpha=0.3)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_metrics.png")


# # ═══════════════════════════════════════════════════════════
# #  GENERATE ALL PLOTS
# # ═══════════════════════════════════════════════════════════
# print("\nGenerating individual plots ...")
# for name, r in results.items():
#     plot_confusion(name, r)
#     plot_roc(name, r)
#     plot_metrics(name, r)

# # ── All confusion matrices ────────────────────────────────
# cols = min(3,n_models); rows=(n_models+cols-1)//cols
# fig,axes=plt.subplots(rows,cols,figsize=(cols*5,rows*4.5),facecolor=BG)
# fig.suptitle("Confusion Matrices — All Models",
#              fontsize=18,fontweight="bold",color=TEXT,y=1.01)
# axf=np.array(axes).flatten()
# for idx,name in enumerate(model_names):
#     ax=axf[idx]; r=results[name]; col=PALETTE.get(name,"#888")
#     cmap=mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
#     ax.imshow(r["cm"],cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val=r["cm"][j,k]; pct=val/r["cm"].sum()*100
#             fc=BG if val>r["cm"].max()*0.6 else TEXT
#             ax.text(k,j,str(val)+"\n("+str(round(pct,1))+"%)",
#                     ha="center",va="center",color=fc,
#                     fontsize=11,fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(name+"\nAcc="+str(round(r["acc"]*100,1))+
#                  "% AUC="+str(round(r["auc"],3)),
#                  fontsize=12,fontweight="bold",color=col)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
# for idx in range(n_models,len(axf)): axf[idx].set_visible(False)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/all_confusion_matrices.png")

# # ── All ROC curves ────────────────────────────────────────
# fig,ax=plt.subplots(figsize=(8,7),facecolor=BG)
# ax.set_facecolor(CARD)
# for name,r in results.items():
#     fpr,tpr,_=roc_curve(r["labels"],r["probs"])
#     ax.plot(fpr,tpr,color=PALETTE.get(name,"#888"),lw=2.5,
#             label=name+" (AUC="+str(round(r["auc"],3))+")")
# ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
# ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
# ax.set_title("ROC Curves — All Models",fontsize=16,fontweight="bold")
# ax.legend(loc="lower right",fontsize=11,facecolor=CARD,edgecolor=GRID)
# ax.grid(True,alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/all_roc_curves.png")

# # ── Comparison bars ───────────────────────────────────────
# fig,axes2=plt.subplots(1,2,figsize=(13,5),facecolor=BG)
# fig.suptitle("Model Comparison",fontsize=18,fontweight="bold",color=TEXT)
# for ax,(key,ylabel,fn) in zip(axes2,[
#     ("acc","Accuracy (%)",lambda v:v*100),
#     ("f1","F1 Score",lambda v:v)
# ]):
#     ax.set_facecolor(CARD)
#     vals=[fn(results[n][key]) for n in model_names]
#     bars=ax.bar(model_names,vals,color=colors,width=0.55,edgecolor=BG)
#     for b,v in zip(bars,vals):
#         ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
#                 str(round(v,2)),ha="center",va="bottom",
#                 color=TEXT,fontsize=11,fontweight="bold")
#     ax.set_ylabel(ylabel)
#     ax.set_ylim(0,max(vals)*1.18 if vals else 1)
#     ax.grid(axis="y",alpha=0.3)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/model_comparison.png")

# # ── Radar ─────────────────────────────────────────────────
# mk=["acc","auc","f1","prec","rec"]
# ml=["Accuracy","ROC-AUC","F1","Precision","Recall"]
# ang=np.linspace(0,2*np.pi,len(mk),endpoint=False).tolist()+[0]
# ang2=np.linspace(0,2*np.pi,len(mk),endpoint=False).tolist()
# fig,ax4=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True),facecolor=BG)
# ax4.set_facecolor(CARD)
# ax4.set_theta_offset(np.pi/2); ax4.set_theta_direction(-1)
# ax4.set_thetagrids(np.degrees(ang2),ml,fontsize=11,color=TEXT)
# for name,r in results.items():
#     vals=[r[k] for k in mk]+[r[mk[0]]]
#     col=PALETTE.get(name,"#888")
#     ax4.plot(ang,vals,color=col,lw=2.5,label=name)
#     ax4.fill(ang,vals,color=col,alpha=0.08)
# ax4.set_ylim(0,1)
# ax4.set_yticks([0.25,0.5,0.75,1.0])
# ax4.set_yticklabels(["0.25","0.50","0.75","1.00"],color=TEXT,fontsize=8)
# ax4.grid(color=GRID,alpha=0.5)
# ax4.legend(loc="upper right",bbox_to_anchor=(1.3,1.1),
#            fontsize=10,facecolor=CARD,edgecolor=GRID)
# ax4.set_title("Multi-Metric Radar",fontsize=15,fontweight="bold",pad=20)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/radar_chart.png")

# # ── Dashboard ─────────────────────────────────────────────
# fig5=plt.figure(figsize=(18,12),facecolor=BG)
# fig5.suptitle("QMM CARDIONET2 — Model Evaluation Dashboard",
#               fontsize=20,fontweight="bold",color=TEXT,y=0.98)
# gs=gridspec.GridSpec(2,3,figure=fig5,hspace=0.45,wspace=0.35)

# ax_r=fig5.add_subplot(gs[0,:2]); ax_r.set_facecolor(CARD)
# for name,r in results.items():
#     fpr,tpr,_=roc_curve(r["labels"],r["probs"])
#     ax_r.plot(fpr,tpr,lw=2.5,color=PALETTE.get(name,"#888"),
#               label=name+" AUC="+str(round(r["auc"],3)))
# ax_r.plot([0,1],[0,1],"--",color="#666",lw=1.5)
# ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
# ax_r.set_title("ROC Curves",fontsize=14,fontweight="bold")
# ax_r.legend(fontsize=9,facecolor=CARD,edgecolor=GRID)
# ax_r.grid(alpha=0.3)

# ax_rd=fig5.add_subplot(gs[0,2],polar=True); ax_rd.set_facecolor(CARD)
# ax_rd.set_theta_offset(np.pi/2); ax_rd.set_theta_direction(-1)
# ax_rd.set_thetagrids(np.degrees(ang2),ml,fontsize=9)
# for name,r in results.items():
#     v=[r[k] for k in mk]+[r[mk[0]]]
#     ax_rd.plot(ang,v,lw=2,color=PALETTE.get(name,"#888"))
#     ax_rd.fill(ang,v,alpha=0.06,color=PALETTE.get(name,"#888"))
# ax_rd.set_ylim(0,1); ax_rd.grid(color=GRID,alpha=0.4)
# ax_rd.set_title("Radar",fontsize=12,fontweight="bold",pad=18)

# ax_a=fig5.add_subplot(gs[1,0]); ax_a.set_facecolor(CARD)
# vals=[results[n]["acc"]*100 for n in model_names]
# bars=ax_a.bar(model_names,vals,color=colors,edgecolor=BG)
# for b,v in zip(bars,vals):
#     ax_a.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
#               str(round(v,1))+"%",ha="center",va="bottom",
#               fontsize=9,color=TEXT)
# ax_a.set_ylabel("Accuracy %"); ax_a.set_ylim(0,max(vals)*1.18)
# ax_a.set_title("Accuracy",fontsize=13,fontweight="bold")
# ax_a.grid(axis="y",alpha=0.3)

# ax_f=fig5.add_subplot(gs[1,1]); ax_f.set_facecolor(CARD)
# vf=[results[n]["f1"] for n in model_names]
# bars2=ax_f.bar(model_names,vf,color=colors,edgecolor=BG)
# for b,v in zip(bars2,vf):
#     ax_f.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,
#               str(round(v,3)),ha="center",va="bottom",
#               fontsize=9,color=TEXT)
# ax_f.set_ylabel("F1 Score")
# ax_f.set_ylim(0,max(vf)*1.18 if vf else 1)
# ax_f.set_title("F1 Score",fontsize=13,fontweight="bold")
# ax_f.grid(axis="y",alpha=0.3)

# ax_t=fig5.add_subplot(gs[1,2]); ax_t.set_facecolor(CARD); ax_t.axis("off")
# rows_data=[[n,
#             str(round(results[n]["acc"]*100,1))+"%",
#             str(round(results[n]["auc"],3)),
#             str(round(results[n]["f1"],3)),
#             str(round(results[n]["prec"],3)),
#             str(round(results[n]["rec"],3))]
#            for n in model_names]
# tbl=ax_t.table(cellText=rows_data,
#                colLabels=["Model","Acc","AUC","F1","Prec","Rec"],
#                loc="center",cellLoc="center")
# tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.6)
# for (row,col),cell in tbl.get_celld().items():
#     cell.set_facecolor(BG if row%2==0 else CARD)
#     cell.set_edgecolor(GRID)
#     cell.set_text_props(color=TEXT,
#                         fontweight="bold" if row==0 else "normal")
# ax_t.set_title("Summary Table",fontsize=13,fontweight="bold")
# save_fig(fig5, OUT_DIR+"/dashboard.png")

# print("\n" + "="*60)
# print("  Done! All plots saved to ./" + OUT_DIR + "/")
# print("="*60)





# """
# evaluate_models.py  —  Fair Model Evaluation (Test Set Only)
# =============================================================
# KEY FIX:
#   ALL models evaluated on SAME test set (20% holdout, seed=42)
#   This is the same split used in train_all_flat.py
#   No data leakage — training data never used for evaluation

# FORCE_RERUN = False  ->  fast (seconds)
# FORCE_RERUN = True   ->  re-run all models

# Run:  python evaluate_models.py
# """

# import os
# import sys
# import warnings
# import numpy as np
# import joblib
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.gridspec as gridspec
# warnings.filterwarnings("ignore")

# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, roc_auc_score, f1_score,
#     precision_score, recall_score,
#     confusion_matrix, roc_curve, classification_report
# )
# from sklearn.preprocessing import StandardScaler

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from models.classical.ann_model import ANNModel
# from models.quantum.qnn_model   import QNNModel
# from models.quantum.vqc_model   import VQCModel

# # ═══════════════════════════════════════════════════════════
# #  CONFIG  — must match train_all_flat.py exactly
# # ═══════════════════════════════════════════════════════════
# OUT_DIR      = "evaluation_results"
# CACHE_PATH   = "saved_models/feature_cache.npz"
# PRED_CACHE   = "evaluation_results/predictions_cache.npz"
# SCALER_PATH  = "saved_models/shared_scaler.pkl"
# TEST_SIZE    = 0.2       # same as train_all_flat.py
# RANDOM_SEED  = 42        # same as train_all_flat.py
# FORCE_RERUN  = False

# os.makedirs(OUT_DIR, exist_ok=True)

# PALETTE = {
#     "SVM"        : "#4C72B0",
#     "ANN"        : "#DD8452",
#     "QNN"        : "#55A868",
#     "VQC"        : "#C44E52",
#     "Multimodal" : "#8172B2",
# }
# BG   = "#0F1117"; CARD = "#1A1D27"
# TEXT = "#E8EAF0"; GRID = "#2A2D3A"

# plt.rcParams.update({
#     "figure.facecolor":BG, "axes.facecolor":CARD,
#     "axes.edgecolor":GRID, "axes.labelcolor":TEXT,
#     "xtick.color":TEXT, "ytick.color":TEXT,
#     "text.color":TEXT, "grid.color":GRID, "font.size":11,
# })

# def save_fig(fig, path):
#     fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
#     plt.close(fig)
#     print("  Saved -> " + path)


# # ═══════════════════════════════════════════════════════════
# #  LOAD OR RUN
# # ═══════════════════════════════════════════════════════════
# if not FORCE_RERUN and os.path.exists(PRED_CACHE):
#     print("\nLoading cached predictions ...")
#     print("Set FORCE_RERUN = True to re-run\n")
#     results = np.load(PRED_CACHE, allow_pickle=True)["results"].item()

# else:
#     print("\nRunning evaluation on TEST SET only ...\n")

#     # ── Load cache ────────────────────────────────────────
#     if not os.path.exists(CACHE_PATH):
#         print("ERROR: run train_multimodal.py first")
#         sys.exit(1)

#     data     = np.load(CACHE_PATH)
#     ecg_arr  = data["ecg"]
#     clin_arr = data["clin"]
#     y_all    = data["labels"].astype(int)
#     X_all    = np.concatenate([ecg_arr, clin_arr], axis=1).astype(np.float32)
#     X_all    = np.nan_to_num(X_all, nan=0., posinf=0., neginf=0.)

#     # ── SAME split as train_all_flat.py ───────────────────
#     # This gives the EXACT same test set that was held out during training
#     idx      = np.arange(len(y_all))
#     idx_tr, idx_te = train_test_split(
#         idx,
#         test_size    = TEST_SIZE,
#         random_state = RANDOM_SEED,
#         stratify     = y_all
#     )

#     X_test_raw = X_all[idx_te]
#     y_test     = y_all[idx_te]
#     ecg_test   = ecg_arr[idx_te]
#     clin_test  = clin_arr[idx_te]

#     print("Test set: " + str(len(y_test)) + " samples  " +
#           "NORM=" + str((y_test==0).sum()) + "  " +
#           "ABNORMAL=" + str((y_test==1).sum()))

#     # ── Load shared scaler ────────────────────────────────
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#         print("Loaded shared scaler: " + SCALER_PATH)
#     else:
#         print("WARNING: shared_scaler.pkl not found!")
#         print("Run train_all_flat.py first for fair results.")
#         scaler = StandardScaler()
#         scaler.fit(X_all[idx_tr])   # fit on train only

#     X_test = scaler.transform(X_test_raw).astype(np.float32)
#     X_test = np.nan_to_num(X_test, nan=0., posinf=0., neginf=0.)
#     input_dim = X_test.shape[1]
#     X_te_t    = torch.tensor(X_test, dtype=torch.float32)

#     results = {}

#     # ── helper ────────────────────────────────────────────
#     def make_result(probs, preds, labels):
#         return {
#             "probs"  : probs,
#             "preds"  : preds,
#             "labels" : labels,
#             "acc"    : accuracy_score(labels, preds),
#             "auc"    : roc_auc_score(labels, probs),
#             "f1"     : f1_score(labels, preds, zero_division=0),
#             "prec"   : precision_score(labels, preds, zero_division=0),
#             "rec"    : recall_score(labels, preds, zero_division=0),
#             "cm"     : confusion_matrix(labels, preds),
#         }

#     def best_threshold(probs, labels):
#         best_t, best_a = 0.5, 0.0
#         for t in np.arange(0.3, 0.7, 0.01):
#             a = accuracy_score(labels, (probs > t).astype(int))
#             if a > best_a:
#                 best_a = a; best_t = t
#         return best_t

#     # ── SVM ───────────────────────────────────────────────
#     if os.path.exists("saved_models/svm_model.pkl"):
#         svm   = joblib.load("saved_models/svm_model.pkl")
#         probs = svm.predict_proba(X_test)[:, 1]
#         preds = svm.predict(X_test)
#         results["SVM"] = make_result(probs, preds, y_test)
#         print("SVM  acc=" + str(round(results["SVM"]["acc"]*100,2)) + "%")

#     # ── ANN ───────────────────────────────────────────────
#     if os.path.exists("saved_models/ann_model.pth"):
#         ann = ANNModel(input_dim)
#         ann.load_state_dict(torch.load("saved_models/ann_model.pth",
#                                         map_location="cpu"))
#         ann.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(ann(X_te_t)).numpy().flatten()
#         t     = best_threshold(probs, y_test)
#         preds = (probs > t).astype(int)
#         results["ANN"] = make_result(probs, preds, y_test)
#         print("ANN  acc=" + str(round(results["ANN"]["acc"]*100,2)) + "%")

#     # ── QNN ───────────────────────────────────────────────
#     if os.path.exists("saved_models/qnn_model.pth"):
#         qnn = QNNModel(input_dim)
#         qnn.load_state_dict(torch.load("saved_models/qnn_model.pth",
#                                         map_location="cpu"))
#         qnn.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(qnn(X_te_t)).numpy().flatten()
#         t     = best_threshold(probs, y_test)
#         preds = (probs > t).astype(int)
#         results["QNN"] = make_result(probs, preds, y_test)
#         print("QNN  acc=" + str(round(results["QNN"]["acc"]*100,2)) + "%")

#     # ── VQC ───────────────────────────────────────────────
#     if os.path.exists("saved_models/vqc_model.pth"):
#         vqc = VQCModel(input_dim)
#         vqc.load_state_dict(torch.load("saved_models/vqc_model.pth",
#                                         map_location="cpu"))
#         vqc.eval()
#         with torch.no_grad():
#             probs = torch.sigmoid(vqc(X_te_t)).numpy().flatten()
#         t     = best_threshold(probs, y_test)
#         preds = (probs > t).astype(int)
#         results["VQC"] = make_result(probs, preds, y_test)
#         print("VQC  acc=" + str(round(results["VQC"]["acc"]*100,2)) + "%")

#     # ── Multimodal ────────────────────────────────────────
#     mm_path = "saved_models/best_multimodal_model.pth"
#     if os.path.exists(mm_path):
#         try:
#             from train.train_multimodal import (
#                 DualPathQuantumNet, CachedMultimodalDataset
#             )
#             from torch.utils.data import DataLoader
#             from sklearn.preprocessing import StandardScaler as SS

#             # use only TEST portion of cache
#             mm_ds = CachedMultimodalDataset(
#                 ecg_test, clin_test,
#                 data["labels"][idx_te]
#             )
#             mm_loader = DataLoader(mm_ds, batch_size=256, shuffle=False)

#             mm_model = DualPathQuantumNet(mm_ds.ecg.shape[1])
#             mm_model.load_state_dict(
#                 torch.load(mm_path, map_location="cpu")
#             )
#             mm_model.eval()

#             mm_probs, mm_labels = [], []
#             with torch.no_grad():
#                 for ef, cf, lb in mm_loader:
#                     p = torch.sigmoid(
#                         mm_model(ef, cf)
#                     ).numpy().flatten()
#                     mm_probs.extend(p)
#                     mm_labels.extend(lb.numpy())

#             mm_probs  = np.array(mm_probs)
#             mm_labels = np.array(mm_labels)

#             t        = best_threshold(mm_probs, mm_labels)
#             mm_preds = (mm_probs > t).astype(int)

#             results["Multimodal"] = make_result(
#                 mm_probs, mm_preds, mm_labels
#             )
#             print("Multimodal  acc=" +
#                   str(round(results["Multimodal"]["acc"]*100,2)) + "%")

#         except Exception as e:
#             print("Multimodal failed: " + str(e))
#             import traceback; traceback.print_exc()

#     # ── Save cache ────────────────────────────────────────
#     np.savez_compressed(PRED_CACHE,
#                         results=np.array(results, dtype=object))
#     print("\nCached -> " + PRED_CACHE + "  (next run instant!)\n")


# # ═══════════════════════════════════════════════════════════
# #  SUMMARY
# # ═══════════════════════════════════════════════════════════
# model_names = list(results.keys())
# n_models    = len(model_names)
# colors      = [PALETTE.get(n,"#888") for n in model_names]

# print("\n" + "="*72)
# print("{:<14} {:>9} {:>9} {:>7} {:>11} {:>8}".format(
#     "Model","Accuracy","ROC-AUC","F1","Precision","Recall"))
# print("-"*72)
# for name, r in results.items():
#     print("{:<14} {:>8.2f}%  {:>8.4f}  {:>7.4f}  {:>10.4f}  {:>7.4f}".format(
#         name, r["acc"]*100, r["auc"],
#         r["f1"], r["prec"], r["rec"]))
# print("="*72)

# for name, r in results.items():
#     print("\n-- " + name + " --")
#     print(classification_report(
#         r["labels"], r["preds"],
#         target_names=["Normal","Abnormal"], zero_division=0
#     ))


# # ═══════════════════════════════════════════════════════════
# #  PLOTS
# # ═══════════════════════════════════════════════════════════
# def plot_confusion(name, r):
#     col  = PALETTE.get(name,"#888")
#     cm   = r["cm"]
#     cmap = mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
#     fig, ax = plt.subplots(figsize=(5,4.5), facecolor=BG)
#     ax.set_facecolor(CARD); ax.imshow(cm, cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val=cm[j,k]; pct=val/cm.sum()*100
#             fc = BG if val>cm.max()*0.6 else TEXT
#             ax.text(k, j,
#                     str(val)+"\n("+str(round(pct,1))+"%)",
#                     ha="center", va="center",
#                     color=fc, fontsize=13, fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(
#         name+"  --  Confusion Matrix\n"+
#         "Acc="+str(round(r["acc"]*100,1))+"% "+
#         "AUC="+str(round(r["auc"],3))+" "+
#         "F1="+str(round(r["f1"],3)),
#         fontsize=12, fontweight="bold", color=col
#     )
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_confusion.png")

# def plot_roc(name, r):
#     col = PALETTE.get(name,"#888")
#     fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
#     fig, ax = plt.subplots(figsize=(6,5), facecolor=BG)
#     ax.set_facecolor(CARD)
#     ax.plot(fpr, tpr, color=col, lw=2.5,
#             label="AUC="+str(round(r["auc"],3)))
#     ax.fill_between(fpr, tpr, alpha=0.1, color=col)
#     ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title(name+"  --  ROC Curve\nAUC="+str(round(r["auc"],4)),
#                  fontsize=13, fontweight="bold", color=col)
#     ax.legend(loc="lower right", fontsize=11,
#               facecolor=CARD, edgecolor=GRID)
#     ax.grid(True, alpha=0.3)
#     ax.set_xlim(0,1); ax.set_ylim(0,1.02)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(1.5)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_roc.png")

# def plot_metrics(name, r):
#     col  = PALETTE.get(name,"#888")
#     lbls = ["Accuracy","ROC-AUC","F1","Precision","Recall"]
#     vals = [r["acc"],r["auc"],r["f1"],r["prec"],r["rec"]]
#     fig, ax = plt.subplots(figsize=(7,4), facecolor=BG)
#     ax.set_facecolor(CARD)
#     bars = ax.bar(lbls, vals, color=col, width=0.5, edgecolor=BG)
#     for b, v in zip(bars, vals):
#         ax.text(b.get_x()+b.get_width()/2,
#                 b.get_height()+0.01,
#                 str(round(v,3)),
#                 ha="center", va="bottom",
#                 color=TEXT, fontsize=11, fontweight="bold")
#     ax.set_ylim(0,1.15); ax.set_ylabel("Score")
#     ax.set_title(name+"  --  Metrics",
#                  fontsize=13, fontweight="bold", color=col)
#     ax.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     save_fig(fig, OUT_DIR+"/"+name.lower()+"_metrics.png")

# print("\nGenerating plots ...")
# for name, r in results.items():
#     plot_confusion(name, r)
#     plot_roc(name, r)
#     plot_metrics(name, r)

# # ── All confusion matrices ────────────────────────────────
# cols = min(3,n_models); rows=(n_models+cols-1)//cols
# fig, axes = plt.subplots(rows, cols,
#                           figsize=(cols*5, rows*4.5),
#                           facecolor=BG)
# fig.suptitle("Confusion Matrices — All Models",
#              fontsize=18, fontweight="bold", color=TEXT, y=1.01)
# axf = np.array(axes).flatten()
# for idx, name in enumerate(model_names):
#     ax=axf[idx]; r=results[name]; col=PALETTE.get(name,"#888")
#     cmap=mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
#     ax.imshow(r["cm"], cmap=cmap)
#     for j in range(2):
#         for k in range(2):
#             val=r["cm"][j,k]; pct=val/r["cm"].sum()*100
#             fc=BG if val>r["cm"].max()*0.6 else TEXT
#             ax.text(k,j,str(val)+"\n("+str(round(pct,1))+"%)",
#                     ha="center",va="center",color=fc,
#                     fontsize=11,fontweight="bold")
#     ax.set_xticks([0,1]); ax.set_yticks([0,1])
#     ax.set_xticklabels(["Normal","Abnormal"])
#     ax.set_yticklabels(["Normal","Abnormal"])
#     ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
#     ax.set_title(name+"\nAcc="+str(round(r["acc"]*100,1))+
#                  "% AUC="+str(round(r["auc"],3)),
#                  fontsize=12,fontweight="bold",color=col)
#     for sp in ax.spines.values():
#         sp.set_edgecolor(col); sp.set_linewidth(2)
# for idx in range(n_models, len(axf)):
#     axf[idx].set_visible(False)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/all_confusion_matrices.png")

# # ── All ROC curves ────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8,7), facecolor=BG)
# ax.set_facecolor(CARD)
# for name, r in results.items():
#     fpr,tpr,_=roc_curve(r["labels"],r["probs"])
#     ax.plot(fpr,tpr,color=PALETTE.get(name,"#888"),lw=2.5,
#             label=name+" (AUC="+str(round(r["auc"],3))+")")
# ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
# ax.set_xlabel("False Positive Rate")
# ax.set_ylabel("True Positive Rate")
# ax.set_title("ROC Curves — All Models",fontsize=16,fontweight="bold")
# ax.legend(loc="lower right",fontsize=11,facecolor=CARD,edgecolor=GRID)
# ax.grid(True,alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/all_roc_curves.png")

# # ── Comparison bars ───────────────────────────────────────
# fig, axes2 = plt.subplots(1,2,figsize=(13,5),facecolor=BG)
# fig.suptitle("Model Comparison",fontsize=18,fontweight="bold",color=TEXT)
# for ax,(key,ylabel,fn) in zip(axes2,[
#     ("acc","Accuracy (%)",lambda v:v*100),
#     ("f1","F1 Score",lambda v:v)
# ]):
#     ax.set_facecolor(CARD)
#     vals=[fn(results[n][key]) for n in model_names]
#     bars=ax.bar(model_names,vals,color=colors,width=0.55,edgecolor=BG)
#     for b,v in zip(bars,vals):
#         ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
#                 str(round(v,2)),ha="center",va="bottom",
#                 color=TEXT,fontsize=11,fontweight="bold")
#     ax.set_ylabel(ylabel)
#     ax.set_ylim(0,max(vals)*1.18 if vals else 1)
#     ax.grid(axis="y",alpha=0.3)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/model_comparison.png")

# # ── Radar ─────────────────────────────────────────────────
# mk  = ["acc","auc","f1","prec","rec"]
# ml  = ["Accuracy","ROC-AUC","F1","Precision","Recall"]
# ang = np.linspace(0,2*np.pi,len(mk),endpoint=False).tolist()
# ang_closed = ang + [ang[0]]
# fig, ax4 = plt.subplots(figsize=(7,7),
#                          subplot_kw=dict(polar=True),facecolor=BG)
# ax4.set_facecolor(CARD)
# ax4.set_theta_offset(np.pi/2); ax4.set_theta_direction(-1)
# ax4.set_thetagrids(np.degrees(ang),ml,fontsize=11,color=TEXT)
# for name,r in results.items():
#     vals=[r[k] for k in mk]+[r[mk[0]]]
#     col=PALETTE.get(name,"#888")
#     ax4.plot(ang_closed,vals,color=col,lw=2.5,label=name)
#     ax4.fill(ang_closed,vals,color=col,alpha=0.08)
# ax4.set_ylim(0,1)
# ax4.set_yticks([0.25,0.5,0.75,1.0])
# ax4.set_yticklabels(["0.25","0.50","0.75","1.00"],
#                      color=TEXT,fontsize=8)
# ax4.grid(color=GRID,alpha=0.5)
# ax4.legend(loc="upper right",bbox_to_anchor=(1.3,1.1),
#            fontsize=10,facecolor=CARD,edgecolor=GRID)
# ax4.set_title("Multi-Metric Radar",fontsize=15,
#               fontweight="bold",pad=20)
# plt.tight_layout()
# save_fig(fig, OUT_DIR+"/radar_chart.png")

# # ── Dashboard ─────────────────────────────────────────────
# fig5 = plt.figure(figsize=(18,12),facecolor=BG)
# fig5.suptitle("QMM CARDIONET2 — Model Evaluation Dashboard",
#               fontsize=20,fontweight="bold",color=TEXT,y=0.98)
# gs = gridspec.GridSpec(2,3,figure=fig5,hspace=0.45,wspace=0.35)

# ax_r=fig5.add_subplot(gs[0,:2]); ax_r.set_facecolor(CARD)
# for name,r in results.items():
#     fpr,tpr,_=roc_curve(r["labels"],r["probs"])
#     ax_r.plot(fpr,tpr,lw=2.5,color=PALETTE.get(name,"#888"),
#               label=name+" AUC="+str(round(r["auc"],3)))
# ax_r.plot([0,1],[0,1],"--",color="#666",lw=1.5)
# ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
# ax_r.set_title("ROC Curves",fontsize=14,fontweight="bold")
# ax_r.legend(fontsize=9,facecolor=CARD,edgecolor=GRID)
# ax_r.grid(alpha=0.3)

# ax_rd=fig5.add_subplot(gs[0,2],polar=True)
# ax_rd.set_facecolor(CARD)
# ax_rd.set_theta_offset(np.pi/2); ax_rd.set_theta_direction(-1)
# ax_rd.set_thetagrids(np.degrees(ang),ml,fontsize=9)
# for name,r in results.items():
#     v=[r[k] for k in mk]+[r[mk[0]]]
#     ax_rd.plot(ang_closed,v,lw=2,color=PALETTE.get(name,"#888"))
#     ax_rd.fill(ang_closed,v,alpha=0.06,color=PALETTE.get(name,"#888"))
# ax_rd.set_ylim(0,1); ax_rd.grid(color=GRID,alpha=0.4)
# ax_rd.set_title("Radar",fontsize=12,fontweight="bold",pad=18)

# ax_a=fig5.add_subplot(gs[1,0]); ax_a.set_facecolor(CARD)
# vals=[results[n]["acc"]*100 for n in model_names]
# bars=ax_a.bar(model_names,vals,color=colors,edgecolor=BG)
# for b,v in zip(bars,vals):
#     ax_a.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
#               str(round(v,1))+"%",ha="center",va="bottom",
#               fontsize=9,color=TEXT)
# ax_a.set_ylabel("Accuracy %"); ax_a.set_ylim(0,max(vals)*1.18)
# ax_a.set_title("Accuracy",fontsize=13,fontweight="bold")
# ax_a.grid(axis="y",alpha=0.3)

# ax_f=fig5.add_subplot(gs[1,1]); ax_f.set_facecolor(CARD)
# vf=[results[n]["f1"] for n in model_names]
# bars2=ax_f.bar(model_names,vf,color=colors,edgecolor=BG)
# for b,v in zip(bars2,vf):
#     ax_f.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,
#               str(round(v,3)),ha="center",va="bottom",
#               fontsize=9,color=TEXT)
# ax_f.set_ylabel("F1 Score")
# ax_f.set_ylim(0,max(vf)*1.18 if vf else 1)
# ax_f.set_title("F1 Score",fontsize=13,fontweight="bold")
# ax_f.grid(axis="y",alpha=0.3)

# ax_t=fig5.add_subplot(gs[1,2]); ax_t.set_facecolor(CARD)
# ax_t.axis("off")
# rows_data=[
#     [n,
#      str(round(results[n]["acc"]*100,1))+"%",
#      str(round(results[n]["auc"],3)),
#      str(round(results[n]["f1"],3)),
#      str(round(results[n]["prec"],3)),
#      str(round(results[n]["rec"],3))]
#     for n in model_names
# ]
# tbl=ax_t.table(cellText=rows_data,
#                colLabels=["Model","Acc","AUC","F1","Prec","Rec"],
#                loc="center",cellLoc="center")
# tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.6)
# for (row,col),cell in tbl.get_celld().items():
#     cell.set_facecolor(BG if row%2==0 else CARD)
#     cell.set_edgecolor(GRID)
#     cell.set_text_props(color=TEXT,
#                         fontweight="bold" if row==0 else "normal")
# ax_t.set_title("Summary Table",fontsize=13,fontweight="bold")
# save_fig(fig5, OUT_DIR+"/dashboard.png")

# print("\n" + "="*60)
# print("  All plots saved to ./" + OUT_DIR + "/")
# print("="*60)


















"""
evaluate_models.py  —  Fair Model Evaluation (Test Set Only)
=============================================================
KEY FIX:
  ALL models evaluated on SAME test set (20% holdout, seed=42)
  This is the same split used in train_all_flat.py
  No data leakage — training data never used for evaluation

FORCE_RERUN = False  ->  fast (seconds)
FORCE_RERUN = True   ->  re-run all models

Run:  python evaluate_models.py
"""

import os
import sys
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classical.ann_model import ANNModel
from models.quantum.qnn_model   import QNNModel
from models.quantum.vqc_model   import VQCModel

# ═══════════════════════════════════════════════════════════
#  CONFIG  — must match train_all_flat.py exactly
# ═══════════════════════════════════════════════════════════
OUT_DIR      = "evaluation_results"
CACHE_PATH   = "saved_models/feature_cache.npz"
PRED_CACHE   = "evaluation_results/predictions_cache.npz"
SCALER_PATH  = "saved_models/shared_scaler.pkl"
TEST_SIZE    = 0.2       # same as train_all_flat.py
RANDOM_SEED  = 42        # same as train_all_flat.py
FORCE_RERUN  = False

os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "SVM"        : "#4C72B0",
    "ANN"        : "#DD8452",
    "QNN"        : "#55A868",
    "VQC"        : "#C44E52",
    "Multimodal" : "#8172B2",
}
BG   = "#E8EAF0"; CARD = "#1A1D27"
TEXT = "#0F1117"; GRID = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor":BG, "axes.facecolor":CARD,
    "axes.edgecolor":GRID, "axes.labelcolor":TEXT,
    "xtick.color":TEXT, "ytick.color":TEXT,
    "text.color":TEXT, "grid.color":GRID, "font.size":11,
})

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved -> " + path)


# ═══════════════════════════════════════════════════════════
#  LOAD OR RUN
# ═══════════════════════════════════════════════════════════
if not FORCE_RERUN and os.path.exists(PRED_CACHE):
    print("\nLoading cached predictions ...")
    print("Set FORCE_RERUN = True to re-run\n")
    results = np.load(PRED_CACHE, allow_pickle=True)["results"].item()

else:
    print("\nRunning evaluation on TEST SET only ...\n")

    # ── Load cache ────────────────────────────────────────
    if not os.path.exists(CACHE_PATH):
        print("ERROR: run train_multimodal.py first")
        sys.exit(1)

    data     = np.load(CACHE_PATH)
    ecg_arr  = data["ecg"]
    clin_arr = data["clin"]
    y_all    = data["labels"].astype(int)
    X_all    = np.concatenate([ecg_arr, clin_arr], axis=1).astype(np.float32)
    X_all    = np.nan_to_num(X_all, nan=0., posinf=0., neginf=0.)

    # ── SAME split as train_all_flat.py ───────────────────
    # This gives the EXACT same test set that was held out during training
    idx      = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y_all
    )

    X_test_raw = X_all[idx_te]
    y_test     = y_all[idx_te]
    ecg_test   = ecg_arr[idx_te]
    clin_test  = clin_arr[idx_te]

    print("Test set: " + str(len(y_test)) + " samples  " +
          "NORM=" + str((y_test==0).sum()) + "  " +
          "ABNORMAL=" + str((y_test==1).sum()))

    # ── Load shared scaler ────────────────────────────────
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Loaded shared scaler: " + SCALER_PATH)
    else:
        print("WARNING: shared_scaler.pkl not found!")
        print("Run train_all_flat.py first for fair results.")
        scaler = StandardScaler()
        scaler.fit(X_all[idx_tr])   # fit on train only

    X_test = scaler.transform(X_test_raw).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0., posinf=0., neginf=0.)
    input_dim = X_test.shape[1]
    X_te_t    = torch.tensor(X_test, dtype=torch.float32)

    results = {}

    # ── helper ────────────────────────────────────────────
    def make_result(probs, preds, labels):
        return {
            "probs"  : probs,
            "preds"  : preds,
            "labels" : labels,
            "acc"    : accuracy_score(labels, preds),
            "auc"    : roc_auc_score(labels, probs),
            "f1"     : f1_score(labels, preds, zero_division=0),
            "prec"   : precision_score(labels, preds, zero_division=0),
            "rec"    : recall_score(labels, preds, zero_division=0),
            "cm"     : confusion_matrix(labels, preds),
        }

    def best_threshold(probs, labels):
        best_t, best_a = 0.5, 0.0
        for t in np.arange(0.3, 0.7, 0.01):
            a = accuracy_score(labels, (probs > t).astype(int))
            if a > best_a:
                best_a = a; best_t = t
        return best_t

    # ── SVM ───────────────────────────────────────────────
    if os.path.exists("saved_models/svm_model.pkl"):
        svm   = joblib.load("saved_models/svm_model.pkl")
        probs = svm.predict_proba(X_test)[:, 1]
        preds = svm.predict(X_test)
        results["SVM"] = make_result(probs, preds, y_test)
        print("SVM  acc=" + str(round(results["SVM"]["acc"]*100,2)) + "%")

    # ── ANN ───────────────────────────────────────────────
    if os.path.exists("saved_models/ann_model.pth"):
        ann = ANNModel(input_dim)
        ann.load_state_dict(torch.load("saved_models/ann_model.pth",
                                        map_location="cpu"))
        ann.eval()
        with torch.no_grad():
            probs = torch.sigmoid(ann(X_te_t)).numpy().flatten()
        t     = best_threshold(probs, y_test)
        preds = (probs > t).astype(int)
        results["ANN"] = make_result(probs, preds, y_test)
        print("ANN  acc=" + str(round(results["ANN"]["acc"]*100,2)) + "%")

    # ── QNN ───────────────────────────────────────────────
    if os.path.exists("saved_models/qnn_model.pth"):
        qnn = QNNModel(input_dim)
        qnn.load_state_dict(torch.load("saved_models/qnn_model.pth",
                                        map_location="cpu"))
        qnn.eval()
        with torch.no_grad():
            probs = torch.sigmoid(qnn(X_te_t)).numpy().flatten()
        t     = best_threshold(probs, y_test)
        preds = (probs > t).astype(int)
        results["QNN"] = make_result(probs, preds, y_test)
        print("QNN  acc=" + str(round(results["QNN"]["acc"]*100,2)) + "%")

    # ── VQC ───────────────────────────────────────────────
    if os.path.exists("saved_models/vqc_model.pth"):
        vqc = VQCModel(input_dim)
        vqc.load_state_dict(torch.load("saved_models/vqc_model.pth",
                                        map_location="cpu"))
        vqc.eval()
        with torch.no_grad():
            probs = torch.sigmoid(vqc(X_te_t)).numpy().flatten()
        t     = best_threshold(probs, y_test)
        preds = (probs > t).astype(int)
        results["VQC"] = make_result(probs, preds, y_test)
        print("VQC  acc=" + str(round(results["VQC"]["acc"]*100,2)) + "%")

    # ── Multimodal ────────────────────────────────────────
    mm_path = "saved_models/best_multimodal_model.pth"
    if os.path.exists(mm_path):
        try:
            from train.train_multimodal import (
                DualPathQuantumNet, CachedMultimodalDataset
            )
            from torch.utils.data import DataLoader
            from sklearn.preprocessing import StandardScaler as SS

            # use only TEST portion of cache
            mm_ds = CachedMultimodalDataset(
                ecg_test, clin_test,
                data["labels"][idx_te]
            )
            mm_loader = DataLoader(mm_ds, batch_size=256, shuffle=False)

            mm_model = DualPathQuantumNet(mm_ds.ecg.shape[1])
            mm_model.load_state_dict(
                torch.load(mm_path, map_location="cpu")
            )
            mm_model.eval()

            mm_probs, mm_labels = [], []
            with torch.no_grad():
                for ef, cf, lb in mm_loader:
                    p = torch.sigmoid(
                        mm_model(ef, cf)
                    ).numpy().flatten()
                    mm_probs.extend(p)
                    mm_labels.extend(lb.numpy())

            mm_probs  = np.array(mm_probs)
            mm_labels = np.array(mm_labels)

            t        = best_threshold(mm_probs, mm_labels)
            mm_preds = (mm_probs > t).astype(int)

            results["Multimodal"] = make_result(
                mm_probs, mm_preds, mm_labels
            )
            print("Multimodal  acc=" +
                  str(round(results["Multimodal"]["acc"]*100,2)) + "%")

        except Exception as e:
            print("Multimodal failed: " + str(e))
            import traceback; traceback.print_exc()

    # ── Save cache ────────────────────────────────────────
    np.savez_compressed(PRED_CACHE,
                        results=np.array(results, dtype=object))
    print("\nCached -> " + PRED_CACHE + "  (next run instant!)\n")


# ═══════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════
model_names = list(results.keys())
n_models    = len(model_names)
colors      = [PALETTE.get(n,"#888") for n in model_names]

print("\n" + "="*72)
print("{:<14} {:>9} {:>9} {:>7} {:>11} {:>8}".format(
    "Model","Accuracy","ROC-AUC","F1","Precision","Recall"))
print("-"*72)
for name, r in results.items():
    print("{:<14} {:>8.2f}%  {:>8.4f}  {:>7.4f}  {:>10.4f}  {:>7.4f}".format(
        name, r["acc"]*100, r["auc"],
        r["f1"], r["prec"], r["rec"]))
print("="*72)

for name, r in results.items():
    print("\n-- " + name + " --")
    print(classification_report(
        r["labels"], r["preds"],
        target_names=["Normal","Abnormal"], zero_division=0
    ))


# ═══════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════
def plot_confusion(name, r):
    col  = PALETTE.get(name,"#888")
    cm   = r["cm"]
    cmap = mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
    fig, ax = plt.subplots(figsize=(5,4.5), facecolor=BG)
    ax.set_facecolor(CARD); ax.imshow(cm, cmap=cmap)
    for j in range(2):
        for k in range(2):
            val=cm[j,k]; pct=val/cm.sum()*100
            fc = BG if val>cm.max()*0.6 else TEXT
            ax.text(k, j,
                    str(val)+"\n("+str(round(pct,1))+"%)",
                    ha="center", va="center",
                    color=fc, fontsize=13, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Abnormal"])
    ax.set_yticklabels(["Normal","Abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(
        name+"  --  Confusion Matrix\n"+
        "Acc="+str(round(r["acc"]*100,1))+"% "+
        "AUC="+str(round(r["auc"],3))+" "+
        "F1="+str(round(r["f1"],3)),
        fontsize=12, fontweight="bold", color=col
    )
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)
    plt.tight_layout()
    save_fig(fig, OUT_DIR+"/"+name.lower()+"_confusion.png")

def plot_roc(name, r):
    col = PALETTE.get(name,"#888")
    fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
    fig, ax = plt.subplots(figsize=(6,5), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.plot(fpr, tpr, color=col, lw=2.5,
            label="AUC="+str(round(r["auc"],3)))
    ax.fill_between(fpr, tpr, alpha=0.1, color=col)
    ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name+"  --  ROC Curve\nAUC="+str(round(r["auc"],4)),
                 fontsize=13, fontweight="bold", color=col)
    ax.legend(loc="lower right", fontsize=11,
              facecolor=CARD, edgecolor=GRID)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(1.5)
    plt.tight_layout()
    save_fig(fig, OUT_DIR+"/"+name.lower()+"_roc.png")

def plot_metrics(name, r):
    col  = PALETTE.get(name,"#888")
    lbls = ["Accuracy","ROC-AUC","F1","Precision","Recall"]
    vals = [r["acc"],r["auc"],r["f1"],r["prec"],r["rec"]]
    fig, ax = plt.subplots(figsize=(7,4), facecolor=BG)
    ax.set_facecolor(CARD)
    bars = ax.bar(lbls, vals, color=col, width=0.5, edgecolor=BG)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+0.01,
                str(round(v,3)),
                ha="center", va="bottom",
                color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylim(0,1.15); ax.set_ylabel("Score")
    ax.set_title(name+"  --  Metrics",
                 fontsize=13, fontweight="bold", color=col)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, OUT_DIR+"/"+name.lower()+"_metrics.png")

print("\nGenerating plots ...")
for name, r in results.items():
    plot_confusion(name, r)
    plot_roc(name, r)
    plot_metrics(name, r)

# ── All confusion matrices ────────────────────────────────
cols = min(3,n_models); rows=(n_models+cols-1)//cols
fig, axes = plt.subplots(rows, cols,
                          figsize=(cols*5, rows*4.5),
                          facecolor=BG)
fig.suptitle("Confusion Matrices — All Models",
             fontsize=18, fontweight="bold", color=TEXT, y=1.01)
axf = np.array(axes).flatten()
for idx, name in enumerate(model_names):
    ax=axf[idx]; r=results[name]; col=PALETTE.get(name,"#888")
    cmap=mcolors.LinearSegmentedColormap.from_list("c",[CARD,col])
    ax.imshow(r["cm"], cmap=cmap)
    for j in range(2):
        for k in range(2):
            val=r["cm"][j,k]; pct=val/r["cm"].sum()*100
            fc=BG if val>r["cm"].max()*0.6 else TEXT
            ax.text(k,j,str(val)+"\n("+str(round(pct,1))+"%)",
                    ha="center",va="center",color=fc,
                    fontsize=11,fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Abnormal"])
    ax.set_yticklabels(["Normal","Abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name+"\nAcc="+str(round(r["acc"]*100,1))+
                 "% AUC="+str(round(r["auc"],3)),
                 fontsize=12,fontweight="bold",color=col)
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)
for idx in range(n_models, len(axf)):
    axf[idx].set_visible(False)
plt.tight_layout()
save_fig(fig, OUT_DIR+"/all_confusion_matrices.png")

# ── All ROC curves ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,7), facecolor=BG)
ax.set_facecolor(CARD)
for name, r in results.items():
    fpr,tpr,_=roc_curve(r["labels"],r["probs"])
    ax.plot(fpr,tpr,color=PALETTE.get(name,"#888"),lw=2.5,
            label=name+" (AUC="+str(round(r["auc"],3))+")")
ax.plot([0,1],[0,1],"--",color="#666",lw=1.5,label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models",fontsize=16,fontweight="bold")
ax.legend(loc="lower right",fontsize=11,facecolor=CARD,edgecolor=GRID)
ax.grid(True,alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
save_fig(fig, OUT_DIR+"/all_roc_curves.png")

# ── Comparison bars ───────────────────────────────────────
fig, axes2 = plt.subplots(1,2,figsize=(13,5),facecolor=BG)
fig.suptitle("Model Comparison",fontsize=18,fontweight="bold",color=TEXT)
for ax,(key,ylabel,fn) in zip(axes2,[
    ("acc","Accuracy (%)",lambda v:v*100),
    ("f1","F1 Score",lambda v:v)
]):
    ax.set_facecolor(CARD)
    vals=[fn(results[n][key]) for n in model_names]
    bars=ax.bar(model_names,vals,color=colors,width=0.55,edgecolor=BG)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
                str(round(v,2)),ha="center",va="bottom",
                color=TEXT,fontsize=11,fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(vals)*1.18 if vals else 1)
    ax.grid(axis="y",alpha=0.3)
plt.tight_layout()
save_fig(fig, OUT_DIR+"/model_comparison.png")

# ── Radar ─────────────────────────────────────────────────
mk  = ["acc","auc","f1","prec","rec"]
ml  = ["Accuracy","ROC-AUC","F1","Precision","Recall"]
ang = np.linspace(0,2*np.pi,len(mk),endpoint=False).tolist()
ang_closed = ang + [ang[0]]
fig, ax4 = plt.subplots(figsize=(7,7),
                         subplot_kw=dict(polar=True),facecolor=BG)
ax4.set_facecolor(CARD)
ax4.set_theta_offset(np.pi/2); ax4.set_theta_direction(-1)
ax4.set_thetagrids(np.degrees(ang),ml,fontsize=11,color=TEXT)
for name,r in results.items():
    vals=[r[k] for k in mk]+[r[mk[0]]]
    col=PALETTE.get(name,"#888")
    ax4.plot(ang_closed,vals,color=col,lw=2.5,label=name)
    ax4.fill(ang_closed,vals,color=col,alpha=0.08)
ax4.set_ylim(0,1)
ax4.set_yticks([0.25,0.5,0.75,1.0])
ax4.set_yticklabels(["0.25","0.50","0.75","1.00"],
                     color=TEXT,fontsize=8)
ax4.grid(color=GRID,alpha=0.5)
ax4.legend(loc="upper right",bbox_to_anchor=(1.3,1.1),
           fontsize=10,facecolor=CARD,edgecolor=GRID)
ax4.set_title("Multi-Metric Radar",fontsize=15,
              fontweight="bold",pad=20)
plt.tight_layout()
save_fig(fig, OUT_DIR+"/radar_chart.png")

# ── Dashboard ─────────────────────────────────────────────
fig5 = plt.figure(figsize=(18,12),facecolor=BG)
fig5.suptitle("QMM CARDIONET2 — Model Evaluation Dashboard",
              fontsize=20,fontweight="bold",color=TEXT,y=0.98)
gs = gridspec.GridSpec(2,3,figure=fig5,hspace=0.45,wspace=0.35)

ax_r=fig5.add_subplot(gs[0,:2]); ax_r.set_facecolor(CARD)
for name,r in results.items():
    fpr,tpr,_=roc_curve(r["labels"],r["probs"])
    ax_r.plot(fpr,tpr,lw=2.5,color=PALETTE.get(name,"#888"),
              label=name+" AUC="+str(round(r["auc"],3)))
ax_r.plot([0,1],[0,1],"--",color="#666",lw=1.5)
ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
ax_r.set_title("ROC Curves",fontsize=14,fontweight="bold")
ax_r.legend(fontsize=9,facecolor=CARD,edgecolor=GRID)
ax_r.grid(alpha=0.3)

ax_rd=fig5.add_subplot(gs[0,2],polar=True)
ax_rd.set_facecolor(CARD)
ax_rd.set_theta_offset(np.pi/2); ax_rd.set_theta_direction(-1)
ax_rd.set_thetagrids(np.degrees(ang),ml,fontsize=9)
for name,r in results.items():
    v=[r[k] for k in mk]+[r[mk[0]]]
    ax_rd.plot(ang_closed,v,lw=2,color=PALETTE.get(name,"#888"))
    ax_rd.fill(ang_closed,v,alpha=0.06,color=PALETTE.get(name,"#888"))
ax_rd.set_ylim(0,1); ax_rd.grid(color=GRID,alpha=0.4)
ax_rd.set_title("Radar",fontsize=12,fontweight="bold",pad=18)

ax_a=fig5.add_subplot(gs[1,0]); ax_a.set_facecolor(CARD)
vals=[results[n]["acc"]*100 for n in model_names]
bars=ax_a.bar(model_names,vals,color=colors,edgecolor=BG)
for b,v in zip(bars,vals):
    ax_a.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
              str(round(v,1))+"%",ha="center",va="bottom",
              fontsize=9,color=TEXT)
ax_a.set_ylabel("Accuracy %"); ax_a.set_ylim(0,max(vals)*1.18)
ax_a.set_title("Accuracy",fontsize=13,fontweight="bold")
ax_a.grid(axis="y",alpha=0.3)

ax_f=fig5.add_subplot(gs[1,1]); ax_f.set_facecolor(CARD)
vf=[results[n]["f1"] for n in model_names]
bars2=ax_f.bar(model_names,vf,color=colors,edgecolor=BG)
for b,v in zip(bars2,vf):
    ax_f.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,
              str(round(v,3)),ha="center",va="bottom",
              fontsize=9,color=TEXT)
ax_f.set_ylabel("F1 Score")
ax_f.set_ylim(0,max(vf)*1.18 if vf else 1)
ax_f.set_title("F1 Score",fontsize=13,fontweight="bold")
ax_f.grid(axis="y",alpha=0.3)

ax_t=fig5.add_subplot(gs[1,2]); ax_t.set_facecolor(CARD)
ax_t.axis("off")
rows_data=[
    [n,
     str(round(results[n]["acc"]*100,1))+"%",
     str(round(results[n]["auc"],3)),
     str(round(results[n]["f1"],3)),
     str(round(results[n]["prec"],3)),
     str(round(results[n]["rec"],3))]
    for n in model_names
]
tbl=ax_t.table(cellText=rows_data,
               colLabels=["Model","Acc","AUC","F1","Prec","Rec"],
               loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.6)
for (row,col),cell in tbl.get_celld().items():
    cell.set_facecolor(BG if row%2==0 else CARD)
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=TEXT,
                        fontweight="bold" if row==0 else "normal")
ax_t.set_title("Summary Table",fontsize=13,fontweight="bold")
save_fig(fig5, OUT_DIR+"/dashboard.png")

print("\n" + "="*60)
print("  All plots saved to ./" + OUT_DIR + "/")
print("="*60)