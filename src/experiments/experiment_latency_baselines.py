"""
experiment_latency_baselines.py — Kiểm định Giả thuyết H2:
"Drift-TKAN phát hiện drift sớm hơn và cho phép thích nghi nhanh hơn 
 các hệ thống cảnh báo hộp đen (ADWIN, PageHinkley, DDM)"

Phương pháp:
- Cho mỗi Detector sử dụng đúng tín hiệu mà nó được thiết kế:
  + Drift-TKAN: Shift Score (representation/distribution drift)
  + ADWIN/PH/DDM: Error stream (performance drift)
- Đo lường: Detection Delay, Max F1 Drop, Recovery Chunks
- Thống kê: Wilcoxon Signed-Rank Test + Cohen's d (Effect Size)
"""

import os
import sys
import logging
import warnings
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from river import drift
except ImportError:
    print("❌ Vui lòng cài đặt river: pip install river")
    sys.exit(1)

from src.utils.experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES, class_names,
    global_scaler, load_all_stream_chunks, get_phase_boundaries,
    load_model_phase1, StreamTimeSeriesDataset, finetune_on_chunk, SEQ_LENGTH
)

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Latency_Experiment')
os.makedirs(REPORTS_DIR, exist_ok=True)
METRICS_CSV = os.path.join(REPORTS_DIR, 'latency_metrics.csv')
SUMMARY_CSV = os.path.join(REPORTS_DIR, 'latency_summary.csv')
PLOT_F1_COMPARISON = os.path.join(REPORTS_DIR, 'f1_comparison.png')

BATCH_SIZE = 512
NUM_WORKERS = 4
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4
EMA_ALPHA = 0.99
SHIFT_THRESHOLD = 0.005  # Ngưỡng cho Drift-TKAN (sẽ được Phase1 calibrate trong production)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [LATENCY] - %(message)s')


def cohens_d(group1, group2):
    """Tính Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-9:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_single_detector(detector_name, all_chunks, device):
    """
    Chạy toàn bộ pipeline luồng liên tục cho 1 detector.
    Trả về: f1_history (list[float]), drift_flags (list[bool])
    """
    model = load_model_phase1(device)
    criterion = nn.CrossEntropyLoss()

    # Khởi tạo detector
    if detector_name == 'ADWIN':
        detector = drift.ADWIN()
    elif detector_name == 'PageHinkley':
        detector = drift.PageHinkley()
    elif detector_name == 'DDM':
        detector = drift.binary.DDM()
    else:
        detector = None  # Drift-TKAN dùng internal Shift Score

    f1_history = []
    drift_flags = []

    for chunk_df, chunk_name in tqdm(all_chunks, desc=f"  {detector_name}", leave=False):
        chunk_ds = StreamTimeSeriesDataset(
            chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler
        )
        chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 1. Prequential Test
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

        # 2. Phát hiện Drift
        drift_detected = False
        if detector_name == 'Drift-TKAN':
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
            drift_detected = (s_kan > SHIFT_THRESHOLD)
        else:
            error_stream = [0 if p == l else 1 for p, l in zip(all_preds, all_labels)]
            for error in error_stream:
                detector.update(error)
                if detector.drift_detected:
                    drift_detected = True
                    break

        drift_flags.append(drift_detected)

        # 3. Fine-tune nếu phát hiện Drift
        if drift_detected:
            model.set_adaptation_mode(is_drifting=True)
            model.train()
            finetune_on_chunk(model, chunk_loader, criterion, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR)
            model.set_adaptation_mode(is_drifting=False)

        # 4. Cập nhật EMA
        model.eval()
        with torch.no_grad():
            for x_val, y_val in chunk_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, context_vec = model(x_val)
                break
        model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)

    return f1_history, drift_flags


def analyze_results(results, all_chunks, phase_boundaries):
    """Phân tích Detection Delay, Max F1 Drop, Recovery Chunks."""
    detector_names = list(results.keys())

    # Ground truth drift boundaries (index đầu tiên khi chuyển phase)
    gt_drift_indices = sorted(phase_boundaries.values())
    logging.info(f"\n📍 Ground-truth drift boundaries (chunk indices): {gt_drift_indices}")

    summary_rows = []

    for det_name in detector_names:
        f1_hist = np.array(results[det_name]['F1'])
        drift_flgs = results[det_name]['Drift']

        # Detection Delay: Từ ground truth, tìm chunk đầu tiên detector cắm cờ SAU mốc đó
        delays = []
        max_f1_drops = []
        recovery_chunks_list = []

        for gt_idx in gt_drift_indices:
            # Baseline F1: trung bình F1 của 5 chunk TRƯỚC mốc drift
            pre_f1 = f1_hist[max(0, gt_idx - 5):gt_idx]
            baseline_f1 = pre_f1.mean() if len(pre_f1) > 0 else f1_hist[0]

            # Detection Delay: chunk đầu tiên sau gt_idx mà detector cắm cờ
            detected_at = None
            for i in range(gt_idx, min(gt_idx + 30, len(drift_flgs))):
                if drift_flgs[i]:
                    detected_at = i
                    break
            delay = (detected_at - gt_idx) if detected_at is not None else 30  # 30 = chưa phát hiện
            delays.append(delay)

            # Max F1 Drop
            post_f1 = f1_hist[gt_idx:min(gt_idx + 30, len(f1_hist))]
            max_drop = baseline_f1 - post_f1.min() if len(post_f1) > 0 else 0
            max_f1_drops.append(max_drop)

            # Recovery Chunks: Chunk đầu tiên F1 >= 90% baseline
            recovery = 30
            for j in range(len(post_f1)):
                if post_f1[j] >= 0.9 * baseline_f1:
                    recovery = j
                    break
            recovery_chunks_list.append(recovery)

        avg_delay = np.mean(delays) if delays else 0
        avg_drop = np.mean(max_f1_drops) if max_f1_drops else 0
        avg_recovery = np.mean(recovery_chunks_list) if recovery_chunks_list else 0
        total_drifts = sum(drift_flgs)

        logging.info(f"\n📊 [{det_name}]")
        logging.info(f"   Avg Detection Delay: {avg_delay:.1f} chunks")
        logging.info(f"   Avg Max F1 Drop:     {avg_drop:.4f}")
        logging.info(f"   Avg Recovery Chunks: {avg_recovery:.1f}")
        logging.info(f"   Total Drifts Found:  {total_drifts}")

        summary_rows.append({
            'Detector': det_name,
            'Avg_Delay': avg_delay,
            'Avg_MaxF1Drop': avg_drop,
            'Avg_Recovery': avg_recovery,
            'Total_Drifts': total_drifts,
            'Delays': delays,
            'F1_Drops': max_f1_drops,
        })

    # Wilcoxon test: Drift-TKAN delays vs mỗi baseline
    tkan_row = [r for r in summary_rows if r['Detector'] == 'Drift-TKAN']
    if tkan_row and len(tkan_row[0]['Delays']) >= 3:
        tkan_delays = np.array(tkan_row[0]['Delays'])
        logging.info(f"\n📊 Statistical Tests (Wilcoxon Signed-Rank):")
        for r in summary_rows:
            if r['Detector'] != 'Drift-TKAN' and len(r['Delays']) == len(tkan_delays):
                baseline_delays = np.array(r['Delays'])
                if not np.array_equal(tkan_delays, baseline_delays):
                    try:
                        stat, p_val = wilcoxon(tkan_delays, baseline_delays)
                        d = cohens_d(tkan_delays, baseline_delays)
                        logging.info(f"   Drift-TKAN vs {r['Detector']}: W={stat:.2f}, p={p_val:.4f}, Cohen's d={d:.3f}")
                    except Exception as e:
                        logging.info(f"   Drift-TKAN vs {r['Detector']}: Không đủ dữ liệu ({e})")

    # Lưu summary CSV
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Detector', 'Avg_Delay', 'Avg_MaxF1Drop', 'Avg_Recovery', 'Total_Drifts'])
        for r in summary_rows:
            writer.writerow([r['Detector'], f"{r['Avg_Delay']:.2f}", f"{r['Avg_MaxF1Drop']:.4f}",
                             f"{r['Avg_Recovery']:.2f}", r['Total_Drifts']])

    return summary_rows


def plot_f1_comparison(results, all_chunks):
    """Vẽ biểu đồ F1 theo thời gian cho tất cả detector."""
    fig, ax = plt.subplots(figsize=(16, 6))
    steps = np.arange(1, len(all_chunks) + 1)

    colors = {'Drift-TKAN': 'tab:red', 'ADWIN': 'tab:blue', 'PageHinkley': 'tab:green', 'DDM': 'tab:orange'}
    for det_name, data in results.items():
        ax.plot(steps[:len(data['F1'])], data['F1'], label=det_name,
                color=colors.get(det_name, 'gray'), linewidth=1.2, alpha=0.85)

    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('F1-Macro')
    ax.set_title('F1-Macro Over Time: Drift-TKAN vs Statistical Baselines')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_F1_COMPARISON, dpi=150)
    logging.info(f"📈 Biểu đồ F1 comparison đã lưu: {PLOT_F1_COMPARISON}")


def run_latency_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Latency Experiment (H2) trên {device}")

    all_chunks = load_all_stream_chunks()
    phase_boundaries = get_phase_boundaries()
    logging.info(f"📂 Tổng cộng {len(all_chunks)} chunks. Phase boundaries: {phase_boundaries}")

    detector_names = ['Drift-TKAN', 'ADWIN', 'PageHinkley', 'DDM']
    results = {}

    for det_name in detector_names:
        logging.info(f"\n{'='*50}")
        logging.info(f"🔬 Chạy pipeline cho: {det_name}")
        logging.info(f"{'='*50}")
        f1_hist, drift_flgs = run_single_detector(det_name, all_chunks, device)
        results[det_name] = {'F1': f1_hist, 'Drift': drift_flgs}

    # Lưu raw metrics
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Chunk_Index', 'Chunk_Name']
        for d in detector_names:
            header.extend([f'{d}_F1', f'{d}_Drift'])
        writer.writerow(header)

        for i, (_, chunk_name) in enumerate(all_chunks):
            row = [i, chunk_name]
            for d in detector_names:
                if i < len(results[d]['F1']):
                    row.extend([f"{results[d]['F1'][i]:.4f}", str(results[d]['Drift'][i])])
                else:
                    row.extend(['N/A', 'N/A'])
            writer.writerow(row)

    # Phân tích và vẽ biểu đồ
    analyze_results(results, all_chunks, phase_boundaries)
    plot_f1_comparison(results, all_chunks)

    logging.info("\n✅ Hoàn thành Latency Experiment (H2).")


if __name__ == "__main__":
    run_latency_experiment()
