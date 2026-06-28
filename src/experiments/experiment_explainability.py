"""
experiment_explainability.py — Kiểm định Giả thuyết H4:
"Drift-TKAN là White-box: có thể giải thích được HỆ SỐ NÀO thay đổi khi Drift xảy ra"

Phương pháp:
- Lưu vết (trajectory) ma trận hệ số Chebyshev qua từng chunk
- Xác định Top-k coefficients thay đổi nhiều nhất khi Drift xảy ra
- Vẽ Heatmap biến động hệ số theo thời gian
- PCA/t-SNE trên không gian hệ số để trực quan hóa cụm Drift vs Stable
"""

import os
import logging
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES,
    global_scaler, load_all_stream_chunks, load_model_phase1,
    StreamTimeSeriesDataset, finetune_on_chunk, SEQ_LENGTH
)

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Explainability_Experiment')
os.makedirs(REPORTS_DIR, exist_ok=True)

HEATMAP_PATH = os.path.join(REPORTS_DIR, 'coefficient_heatmap.png')
PCA_PATH = os.path.join(REPORTS_DIR, 'coefficient_pca.png')
TOPK_CSV = os.path.join(REPORTS_DIR, 'topk_coefficients.csv')
TRAJECTORY_CSV = os.path.join(REPORTS_DIR, 'coefficient_trajectory.csv')

BATCH_SIZE = 512
NUM_WORKERS = 4
SHIFT_THRESHOLD = 0.005

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EXPLAIN] - %(message)s')


def run_explainability_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Explainability Experiment (H4) trên {device}")

    model = load_model_phase1(device)
    criterion = nn.CrossEntropyLoss()

    all_chunks = load_all_stream_chunks()
    logging.info(f"📂 Tổng cộng {len(all_chunks)} chunks.")

    # Lưu snapshot ban đầu của Chebyshev coefficients
    initial_coeffs = model.kan_classifier.cheb_coeffs.detach().cpu().numpy().flatten()
    coeff_dim = len(initial_coeffs)
    logging.info(f"📐 Số chiều hệ số Chebyshev: {coeff_dim}")

    trajectory = []  # List of (chunk_name, coeffs_flat, shift_score, is_drift)
    chunk_labels = []

    for step, (chunk_df, chunk_name) in enumerate(all_chunks):
        chunk_ds = StreamTimeSeriesDataset(
            chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler
        )
        chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Light Adaptation
        model.set_adaptation_mode(is_drifting=True)
        finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)

        # Measure Drift
        model.eval()
        with torch.no_grad():
            for x_val, _ in chunk_loader:
                x_val = x_val.to(device)
                logits, context_vec = model(x_val)
                _, s_kan = model.compute_hierarchical_drift(context_vec, logits)
                break

        is_drift = s_kan > SHIFT_THRESHOLD
        current_coeffs = model.kan_classifier.cheb_coeffs.detach().cpu().numpy().flatten()
        trajectory.append((chunk_name, current_coeffs.copy(), s_kan, is_drift))
        chunk_labels.append('Drift' if is_drift else 'Stable')

        model.set_adaptation_mode(is_drifting=False)

        # Update EMA
        with torch.no_grad():
            for x_val, y_val in chunk_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, context_vec = model(x_val)
                break
        model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=0.99)

        if (step + 1) % 50 == 0:
            logging.info(f"  [{step+1}/{len(all_chunks)}] S_KAN={s_kan:.6f} {'🔴 DRIFT' if is_drift else '🟢'}")

    # ==================== PHÂN TÍCH ====================
    logging.info("\n📊 Phân tích Explainability...")

    all_coeffs = np.array([t[1] for t in trajectory])  # [N_chunks, coeff_dim]
    all_shifts = np.array([t[2] for t in trajectory])
    all_drift_flags = np.array([t[3] for t in trajectory])

    # --- 1. Top-k coefficients thay đổi nhiều nhất ---
    coeff_diff = np.abs(all_coeffs - initial_coeffs)  # So với trạng thái ban đầu
    mean_change = coeff_diff.mean(axis=0)  # Trung bình qua tất cả chunks
    topk = 20
    top_indices = np.argsort(mean_change)[::-1][:topk]

    logging.info(f"\n🔍 Top-{topk} hệ số Chebyshev thay đổi nhiều nhất:")
    with open(TOPK_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Coeff_Index', 'Mean_Abs_Change', 'Std_Change'])
        for rank, idx in enumerate(top_indices):
            std_change = coeff_diff[:, idx].std()
            logging.info(f"  #{rank+1}: Coeff[{idx}] | Mean Δ = {mean_change[idx]:.6f} | Std = {std_change:.6f}")
            writer.writerow([rank+1, idx, f"{mean_change[idx]:.6f}", f"{std_change:.6f}"])

    # --- 2. Heatmap ---
    fig, ax = plt.subplots(figsize=(16, 8))
    # Chỉ vẽ top-50 coefficients để dễ đọc
    top50 = np.argsort(mean_change)[::-1][:50]
    heatmap_data = coeff_diff[:, top50].T  # [50, N_chunks]

    im = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Coefficient Index (Top-50)')
    ax.set_title('Chebyshev Coefficient Change Heatmap (|Current - Initial|)')
    plt.colorbar(im, ax=ax, label='Absolute Change')

    # Đánh dấu các chunk có Drift
    drift_chunks = np.where(all_drift_flags)[0]
    for dc in drift_chunks:
        ax.axvline(x=dc, color='cyan', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150)
    logging.info(f"📈 Heatmap đã lưu: {HEATMAP_PATH}")

    # --- 3. PCA ---
    if len(all_coeffs) > 5:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_coeffs)

        fig, ax = plt.subplots(figsize=(10, 8))
        stable_mask = ~all_drift_flags
        drift_mask = all_drift_flags

        ax.scatter(coords[stable_mask, 0], coords[stable_mask, 1],
                   c='steelblue', alpha=0.5, s=20, label='Stable')
        ax.scatter(coords[drift_mask, 0], coords[drift_mask, 1],
                   c='red', alpha=0.7, s=40, marker='x', label='Drift')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA of Chebyshev Coefficient Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PCA_PATH, dpi=150)
        logging.info(f"📈 PCA plot đã lưu: {PCA_PATH}")

    # --- 4. Lưu trajectory CSV ---
    with open(TRAJECTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Chunk', 'Shift_Score', 'Is_Drift'] + [f'Coeff_{i}' for i in range(min(50, coeff_dim))]
        writer.writerow(header)
        for name, coeffs, shift, is_d in trajectory:
            row = [name, f"{shift:.6f}", str(is_d)] + [f"{c:.6f}" for c in coeffs[:50]]
            writer.writerow(row)

    logging.info("\n✅ Hoàn thành Explainability Experiment (H4).")


if __name__ == "__main__":
    run_explainability_experiment()
