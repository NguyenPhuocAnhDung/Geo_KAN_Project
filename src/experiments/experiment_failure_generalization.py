"""
experiment_failure_generalization.py — Kiểm định Giả thuyết H6:
"Drift-TKAN ổn định dưới nhiều điều kiện và không phụ thuộc thứ tự dữ liệu"

Phần 1 - Failure Analysis:
- Bơm Gaussian noise vào features
- Bơm label noise (đảo nhãn ngẫu nhiên)
- Đo False Alarm Rate và Mean Time Between False Alarms (MTBFA)

Phần 2 - Generalization:
- Đảo thứ tự các Dataset (Phase2/Phase3)
- Chạy nhiều lần (Run A, B, C) với thứ tự khác nhau
- So sánh xu hướng F1 để xem mô hình có phụ thuộc thứ tự không
"""

import os
import logging
import warnings
import csv
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES,
    global_scaler, load_phase_chunks, load_model_phase1,
    StreamTimeSeriesDataset, finetune_on_chunk, SEQ_LENGTH
)

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Failure_Generalization')
os.makedirs(REPORTS_DIR, exist_ok=True)
FAILURE_CSV = os.path.join(REPORTS_DIR, 'failure_analysis.csv')
GENERAL_CSV = os.path.join(REPORTS_DIR, 'generalization_metrics.csv')
GENERAL_PLOT = os.path.join(REPORTS_DIR, 'generalization_comparison.png')

BATCH_SIZE = 512
NUM_WORKERS = 4
EMA_ALPHA = 0.99
SHIFT_THRESHOLD = 0.005

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [FAILURE] - %(message)s')


# ==================== PHẦN 1: FAILURE ANALYSIS ====================

def inject_gaussian_noise(chunk_df, noise_std=0.5):
    """Bơm Gaussian noise vào features (không ảnh hưởng Label)."""
    df_noisy = chunk_df.copy()
    feature_cols = [c for c in df_noisy.columns if c != 'Label']
    noise = np.random.normal(0, noise_std, size=(len(df_noisy), len(feature_cols)))
    df_noisy[feature_cols] = df_noisy[feature_cols].values + noise
    return df_noisy


def inject_label_noise(chunk_df, flip_rate=0.2):
    """Đảo nhãn ngẫu nhiên một tỷ lệ mẫu."""
    df_noisy = chunk_df.copy()
    if pd.api.types.is_categorical_dtype(df_noisy['Label']):
        df_noisy['Label'] = df_noisy['Label'].astype(str)
    n_flip = int(len(df_noisy) * flip_rate)
    flip_indices = np.random.choice(len(df_noisy), n_flip, replace=False)
    all_labels = list(GLOBAL_LABEL_MAP.keys())
    for idx in flip_indices:
        current_label = df_noisy.iloc[idx]['Label']
        other_labels = [l for l in all_labels if l != current_label]
        if other_labels:
            df_noisy.iloc[idx, df_noisy.columns.get_loc('Label')] = np.random.choice(other_labels)
    return df_noisy


def run_failure_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("🔬 Phần 1: Failure Analysis")

    # Lấy 1 sample nhỏ từ Phase2
    phase2_chunks = load_phase_chunks("Phase2_DriftTest", fraction=0.05)
    if not phase2_chunks:
        logging.error("❌ Không có dữ liệu Phase2.")
        return

    # Chỉ lấy 50 chunks đầu để test nhanh
    test_chunks = phase2_chunks[:50]

    scenarios = [
        ('Clean', None),
        ('Gaussian_Noise_0.1', lambda df: inject_gaussian_noise(df, 0.1)),
        ('Gaussian_Noise_0.5', lambda df: inject_gaussian_noise(df, 0.5)),
        ('Gaussian_Noise_1.0', lambda df: inject_gaussian_noise(df, 1.0)),
        ('Label_Noise_10%', lambda df: inject_label_noise(df, 0.1)),
        ('Label_Noise_20%', lambda df: inject_label_noise(df, 0.2)),
    ]

    with open(FAILURE_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Total_Chunks', 'Total_Drifts', 'False_Alarm_Rate', 'MTBFA', 'Avg_F1'])

    for scenario_name, transform_fn in scenarios:
        logging.info(f"\n  🧪 Kịch bản: {scenario_name}")

        model = load_model_phase1(device)
        criterion = nn.CrossEntropyLoss()

        drift_count = 0
        chunks_since_last_alarm = 0
        mtbfa_intervals = []
        f1_list = []

        for chunk_df, chunk_name in tqdm(test_chunks, desc=f"    {scenario_name}", leave=False):
            # Áp dụng biến đổi (nếu có)
            if transform_fn is not None:
                chunk_df = transform_fn(chunk_df)

            chunk_ds = StreamTimeSeriesDataset(
                chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler
            )
            chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # Prequential Test
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in chunk_loader:
                    x, y = x.to(device), y.to(device)
                    x = torch.nan_to_num(x, nan=0.0)
                    logits, _ = model(x)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_list.append(f1)

            # Drift Detection
            model.set_adaptation_mode(is_drifting=True)
            finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)
            model.eval()
            with torch.no_grad():
                for x_val, _ in chunk_loader:
                    x_val = x_val.to(device)
                    logits, context_vec = model(x_val)
                    _, s_kan = model.compute_hierarchical_drift(context_vec, logits)
                    break
            model.set_adaptation_mode(is_drifting=False)

            chunks_since_last_alarm += 1
            if s_kan > SHIFT_THRESHOLD:
                drift_count += 1
                if chunks_since_last_alarm > 0:
                    mtbfa_intervals.append(chunks_since_last_alarm)
                chunks_since_last_alarm = 0

            # Update EMA
            with torch.no_grad():
                for x_val, y_val in chunk_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    _, context_vec = model(x_val)
                    break
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)

        far = drift_count / len(test_chunks)
        mtbfa = np.mean(mtbfa_intervals) if mtbfa_intervals else len(test_chunks)
        avg_f1 = np.mean(f1_list)

        logging.info(f"    Drifts={drift_count} | FAR={far:.3f} | MTBFA={mtbfa:.1f} | Avg F1={avg_f1:.4f}")

        with open(FAILURE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, len(test_chunks), drift_count,
                             f"{far:.4f}", f"{mtbfa:.2f}", f"{avg_f1:.4f}"])


# ==================== PHẦN 2: GENERALIZATION ====================

def run_generalization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("\n🔬 Phần 2: Generalization (Đảo thứ tự Dataset)")

    # Tải chunks từ từng Phase riêng
    phase2_chunks = load_phase_chunks("Phase2_DriftTest", fraction=0.05)
    phase3_chunks = load_phase_chunks("Phase3_DomainShift", fraction=0.05)

    # Định nghĩa các Run với thứ tự khác nhau
    runs = {
        'Run_A (P2→P3)': phase2_chunks + phase3_chunks,
        'Run_B (P3→P2)': phase3_chunks + phase2_chunks,
    }

    with open(GENERAL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Chunk_Index', 'F1_Macro'])

    all_f1_histories = {}

    for run_name, chunks in runs.items():
        logging.info(f"\n  🔄 {run_name}: {len(chunks)} chunks")

        model = load_model_phase1(device)
        criterion = nn.CrossEntropyLoss()
        f1_history = []

        for chunk_df, chunk_name in tqdm(chunks, desc=f"    {run_name}", leave=False):
            chunk_ds = StreamTimeSeriesDataset(
                chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler
            )
            chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # Prequential
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in chunk_loader:
                    x, y = x.to(device), y.to(device)
                    x = torch.nan_to_num(x, nan=0.0)
                    logits, _ = model(x)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_history.append(f1)

            # Light adapt + drift check + finetune
            model.set_adaptation_mode(is_drifting=True)
            finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)
            model.eval()
            with torch.no_grad():
                for x_val, _ in chunk_loader:
                    x_val = x_val.to(device)
                    logits, context_vec = model(x_val)
                    _, s_kan = model.compute_hierarchical_drift(context_vec, logits)
                    break

            if s_kan > SHIFT_THRESHOLD:
                model.train()
                finetune_on_chunk(model, chunk_loader, criterion, epochs=3, lr=1e-4)

            model.set_adaptation_mode(is_drifting=False)
            with torch.no_grad():
                for x_val, y_val in chunk_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    _, context_vec = model(x_val)
                    break
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)

        all_f1_histories[run_name] = f1_history
        avg_f1 = np.mean(f1_history)
        logging.info(f"    Avg F1: {avg_f1:.4f}")

        with open(GENERAL_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, f1_val in enumerate(f1_history):
                writer.writerow([run_name, i, f"{f1_val:.4f}"])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'Run_A (P2→P3)': 'tab:blue', 'Run_B (P3→P2)': 'tab:red'}
    for run_name, f1_hist in all_f1_histories.items():
        steps = np.arange(1, len(f1_hist) + 1)
        ax.plot(steps, f1_hist, label=run_name, color=colors.get(run_name, 'gray'),
                linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('F1-Macro')
    ax.set_title('Generalization Test: F1 under Different Dataset Orderings')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GENERAL_PLOT, dpi=150)
    logging.info(f"📈 Generalization plot đã lưu: {GENERAL_PLOT}")


def run_experiment():
    logging.info("🚀 Khởi chạy Failure & Generalization Experiment (H6)")
    run_failure_analysis()
    run_generalization()
    logging.info("\n✅ Hoàn thành Failure & Generalization Experiment (H6).")


if __name__ == "__main__":
    run_experiment()
