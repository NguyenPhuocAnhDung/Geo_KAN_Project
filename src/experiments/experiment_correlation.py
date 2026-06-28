"""
experiment_correlation.py — Kiểm định Giả thuyết H1:
"Chebyshev coefficients mang thông tin về concept drift"

Phương pháp:
- Thu thập Shift Score, F1, Accuracy, Loss, ECE, Entropy trên từng chunk
- Tính Pearson / Spearman correlation
- Tính Cross-Correlation để chứng minh Shift Score là tín hiệu Early Warning
"""

import os
import logging
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES, class_names,
    global_scaler, load_all_stream_chunks, load_model_phase1,
    StreamTimeSeriesDataset, finetune_on_chunk, SEQ_LENGTH
)

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Correlation_Experiment')
os.makedirs(REPORTS_DIR, exist_ok=True)
METRICS_CSV = os.path.join(REPORTS_DIR, 'correlation_metrics.csv')
PLOT_CROSS_CORR = os.path.join(REPORTS_DIR, 'cross_correlation_plot.png')
PLOT_TIMESERIES = os.path.join(REPORTS_DIR, 'timeseries_overlay.png')

BATCH_SIZE = 512
NUM_WORKERS = 4
EMA_ALPHA = 0.99

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CORRELATION] - %(message)s')


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Tính ECE (Expected Calibration Error)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].astype(float).mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return ece


def run_correlation_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Correlation Experiment (H1) trên {device}")

    # Load model Phase 1
    model = load_model_phase1(device)
    logging.info("✅ Đã load mô hình Phase 1.")

    criterion = nn.CrossEntropyLoss()

    # Load tất cả stream chunks
    all_chunks = load_all_stream_chunks()
    logging.info(f"📂 Tổng cộng {len(all_chunks)} chunks từ Phase2 + Phase3.")

    # Khởi tạo CSV
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Chunk', 'Shift_Score_KAN', 'Shift_Score_Rep',
                         'F1_Macro', 'Accuracy', 'Loss', 'Entropy', 'ECE', 'Confidence'])

    metrics = {k: [] for k in ['Shift_KAN', 'Shift_Rep', 'F1', 'Acc', 'Loss', 'Entropy', 'ECE', 'Confidence']}

    for step, (chunk_df, chunk_name) in enumerate(all_chunks):
        logging.info(f"🔄 [{step+1}/{len(all_chunks)}] Chunk: {chunk_name}")

        chunk_ds = StreamTimeSeriesDataset(chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # --- Bước 1: Prequential Test (đo hiệu năng TRƯỚC khi adapt) ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss, total_entropy, num_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in chunk_loader:
                x, y = x.to(device), y.to(device)
                x = torch.nan_to_num(x, nan=0.0)
                logits, _ = model(x)

                loss = criterion(logits, y)
                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean().item()
                total_entropy += entropy

                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                num_batches += 1

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / max(num_batches, 1)
        avg_entropy = total_entropy / max(num_batches, 1)
        ece = expected_calibration_error(np.array(all_labels), np.array(all_probs))
        avg_confidence = np.max(np.array(all_probs), axis=1).mean()

        # --- Bước 2: Light Adaptation → Đo Shift Score ---
        model.set_adaptation_mode(is_drifting=True)
        finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)

        model.eval()
        with torch.no_grad():
            for x_val, y_val in chunk_loader:
                x_val = x_val.to(device)
                logits, context_vec = model(x_val)
                s_rep, s_kan = model.compute_hierarchical_drift(context_vec, logits)
                break

        model.set_adaptation_mode(is_drifting=False)

        # Cập nhật EMA
        with torch.no_grad():
            for x_val, y_val in chunk_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, context_vec = model(x_val)
                break
        model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)

        logging.info(f"  F1={f1:.4f} | ECE={ece:.4f} | S_KAN={s_kan:.6f} | S_Rep={s_rep:.6f}")

        # Lưu metrics
        with open(METRICS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step+1, chunk_name, f"{s_kan:.6f}", f"{s_rep:.6f}",
                             f"{f1:.4f}", f"{acc:.4f}", f"{avg_loss:.4f}",
                             f"{avg_entropy:.4f}", f"{ece:.4f}", f"{avg_confidence:.4f}"])

        metrics['Shift_KAN'].append(s_kan)
        metrics['Shift_Rep'].append(s_rep)
        metrics['F1'].append(f1)
        metrics['Acc'].append(acc)
        metrics['Loss'].append(avg_loss)
        metrics['Entropy'].append(avg_entropy)
        metrics['ECE'].append(ece)
        metrics['Confidence'].append(avg_confidence)

    # ==================== PHÂN TÍCH TƯƠNG QUAN ====================
    logging.info("\n" + "=" * 60)
    logging.info("📊 PHÂN TÍCH TƯƠNG QUAN (CORRELATION ANALYSIS)")
    logging.info("=" * 60)

    shift = np.array(metrics['Shift_KAN'])
    f1_arr = np.array(metrics['F1'])
    ece_arr = np.array(metrics['ECE'])
    entropy_arr = np.array(metrics['Entropy'])

    if len(shift) > 3:
        pairs = [
            ('Shift_KAN', 'F1', shift, f1_arr),
            ('Shift_KAN', 'ECE', shift, ece_arr),
            ('Shift_KAN', 'Entropy', shift, entropy_arr),
        ]
        for name_a, name_b, arr_a, arr_b in pairs:
            pr, pp = pearsonr(arr_a, arr_b)
            sr, sp = spearmanr(arr_a, arr_b)
            logging.info(f"  Pearson  ({name_a} vs {name_b}): r = {pr:.4f}, p = {pp:.2e}")
            logging.info(f"  Spearman ({name_a} vs {name_b}): ρ = {sr:.4f}, p = {sp:.2e}")

        # Cross-Correlation (Shift vs F1)
        shift_norm = (shift - shift.mean()) / (shift.std() + 1e-9)
        f1_norm = (f1_arr - f1_arr.mean()) / (f1_arr.std() + 1e-9)
        cross_corr = np.correlate(shift_norm, f1_norm, mode='full')
        lags = np.arange(-len(shift) + 1, len(shift))
        peak_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[peak_idx]

        logging.info(f"\n  ⏱️ Cross-Correlation Optimal Lag: {optimal_lag} chunks")
        if optimal_lag < 0:
            logging.info("  => Shift Score phản ứng TRƯỚC F1 tụt (Early Warning confirmed!)")
        elif optimal_lag == 0:
            logging.info("  => Shift Score và F1 thay đổi ĐỒNG THỜI")
        else:
            logging.info("  => Shift Score phản ứng SAU F1 (không phải early warning)")

        # --- Vẽ biểu đồ Cross-Correlation ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(lags, cross_corr, color='steelblue', linewidth=1.5)
        ax.axvline(x=optimal_lag, color='red', linestyle='--', label=f'Optimal Lag = {optimal_lag}')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Lag (Chunks)')
        ax.set_ylabel('Cross-Correlation')
        ax.set_title('Cross-Correlation: Shift Score (KAN) vs F1-Macro')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_CROSS_CORR, dpi=150)
        logging.info(f"  📈 Biểu đồ Cross-Correlation đã lưu: {PLOT_CROSS_CORR}")

        # --- Vẽ biểu đồ Time-series Overlay ---
        fig, ax1 = plt.subplots(figsize=(14, 6))
        steps = np.arange(1, len(shift) + 1)

        ax1.set_xlabel('Chunk Index')
        ax1.set_ylabel('F1-Macro', color='tab:blue')
        ax1.plot(steps, f1_arr, color='tab:blue', linewidth=1.5, label='F1-Macro', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Shift Score (KAN)', color='tab:red')
        ax2.plot(steps, shift, color='tab:red', linewidth=1.5, label='Shift Score', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.suptitle('Time-Series Overlay: F1-Macro vs KAN Shift Score', fontsize=13)
        fig.tight_layout()
        plt.savefig(PLOT_TIMESERIES, dpi=150)
        logging.info(f"  📈 Biểu đồ Time-series đã lưu: {PLOT_TIMESERIES}")

    logging.info("\n✅ Hoàn thành Correlation Experiment (H1).")


if __name__ == "__main__":
    run_correlation_experiment()
