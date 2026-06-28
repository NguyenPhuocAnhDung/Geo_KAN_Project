"""
experiment_complexity.py — Kiểm định Giả thuyết H5:
"Chi phí tính toán của Drift Sensor là chấp nhận được"

Phương pháp:
- Tách bạch Model Complexity vs Detector Overhead
- Model Complexity: Parameters, FLOPs (ước tính), Inference time
- Detector Overhead: CPU/GPU time cho Shift Score vs ADWIN/DDM/PH
"""

import os
import sys
import time
import logging
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from river import drift
except ImportError:
    print("❌ Vui lòng cài đặt river: pip install river")
    sys.exit(1)

from experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES,
    global_scaler, load_all_stream_chunks, load_model_phase1,
    StreamTimeSeriesDataset, SEQ_LENGTH
)

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Complexity_Experiment')
os.makedirs(REPORTS_DIR, exist_ok=True)
MODEL_COMPLEXITY_CSV = os.path.join(REPORTS_DIR, 'model_complexity.csv')
DETECTOR_OVERHEAD_CSV = os.path.join(REPORTS_DIR, 'detector_overhead.csv')

BATCH_SIZE = 512
NUM_WORKERS = 4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [COMPLEXITY] - %(message)s')


def count_parameters(model):
    """Đếm tổng số tham số (total, trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size_mb(model):
    """Ước tính kích thước mô hình (MB)."""
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total_bytes + buffer_bytes) / (1024 ** 2)


def measure_inference_time(model, device, n_runs=100):
    """Đo thời gian inference trung bình (ms)."""
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, len(GLOBAL_FEATURE_COLS)).to(device)
    model.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def measure_shift_score_overhead(model, device, n_runs=100):
    """Đo thời gian tính Shift Score (ms)."""
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, len(GLOBAL_FEATURE_COLS)).to(device)
    model.eval()

    with torch.no_grad():
        logits, context_vec = model(dummy_input)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.compute_hierarchical_drift(context_vec, logits)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def measure_adwin_overhead(n_samples=10000, n_runs=10):
    """Đo thời gian xử lý luồng lỗi bằng ADWIN (ms)."""
    times = []
    for _ in range(n_runs):
        detector = drift.ADWIN()
        errors = np.random.binomial(1, 0.1, size=n_samples)
        start = time.perf_counter()
        for e in errors:
            detector.update(e)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


def measure_ddm_overhead(n_samples=10000, n_runs=10):
    """Đo thời gian xử lý luồng lỗi bằng DDM (ms)."""
    times = []
    for _ in range(n_runs):
        detector = drift.binary.DDM()
        errors = np.random.binomial(1, 0.1, size=n_samples)
        start = time.perf_counter()
        for e in errors:
            detector.update(e)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


def run_complexity_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Complexity Experiment (H5) trên {device}")

    # ==================== MODEL COMPLEXITY ====================
    logging.info("\n" + "=" * 50)
    logging.info("📐 MODEL COMPLEXITY")
    logging.info("=" * 50)

    # Full model (BiLSTM + KAN)
    model_full = load_model_phase1(device)
    total_full, trainable_full = count_parameters(model_full)
    size_full = estimate_model_size_mb(model_full)
    infer_mean_full, infer_std_full = measure_inference_time(model_full, device)

    # Variant: Replace KAN with Linear
    model_linear = load_model_phase1(device)
    in_f = model_linear.kan_classifier.in_features
    out_f = model_linear.kan_classifier.out_features
    model_linear.kan_classifier = nn.Linear(in_f, out_f).to(device)
    total_linear, trainable_linear = count_parameters(model_linear)
    size_linear = estimate_model_size_mb(model_linear)
    infer_mean_linear, infer_std_linear = measure_inference_time(model_linear, device)

    logging.info(f"\n  BiLSTM+KAN:    Params={total_full:,} | Size={size_full:.2f}MB | "
                 f"Inference={infer_mean_full:.2f}±{infer_std_full:.2f}ms")
    logging.info(f"  BiLSTM+Linear: Params={total_linear:,} | Size={size_linear:.2f}MB | "
                 f"Inference={infer_mean_linear:.2f}±{infer_std_linear:.2f}ms")
    logging.info(f"  KAN Overhead:  Params=+{total_full-total_linear:,} | "
                 f"Size=+{size_full-size_linear:.2f}MB | "
                 f"Inference=+{infer_mean_full-infer_mean_linear:.2f}ms")

    with open(MODEL_COMPLEXITY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Total_Params', 'Trainable_Params', 'Size_MB',
                         'Inference_Mean_ms', 'Inference_Std_ms'])
        writer.writerow(['BiLSTM+KAN', total_full, trainable_full, f"{size_full:.3f}",
                         f"{infer_mean_full:.3f}", f"{infer_std_full:.3f}"])
        writer.writerow(['BiLSTM+Linear', total_linear, trainable_linear, f"{size_linear:.3f}",
                         f"{infer_mean_linear:.3f}", f"{infer_std_linear:.3f}"])

    # ==================== DETECTOR OVERHEAD ====================
    logging.info("\n" + "=" * 50)
    logging.info("⚡ DETECTOR OVERHEAD")
    logging.info("=" * 50)

    shift_mean, shift_std = measure_shift_score_overhead(model_full, device)
    adwin_mean, adwin_std = measure_adwin_overhead()
    ddm_mean, ddm_std = measure_ddm_overhead()

    logging.info(f"\n  Shift Score (KAN): {shift_mean:.3f}±{shift_std:.3f}ms per batch")
    logging.info(f"  ADWIN (10k samples): {adwin_mean:.3f}±{adwin_std:.3f}ms")
    logging.info(f"  DDM (10k samples):   {ddm_mean:.3f}±{ddm_std:.3f}ms")

    with open(DETECTOR_OVERHEAD_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Detector', 'Mean_ms', 'Std_ms', 'Notes'])
        writer.writerow(['Shift_Score_KAN', f"{shift_mean:.3f}", f"{shift_std:.3f}", 'Per batch (512 samples)'])
        writer.writerow(['ADWIN', f"{adwin_mean:.3f}", f"{adwin_std:.3f}", '10k error samples'])
        writer.writerow(['DDM', f"{ddm_mean:.3f}", f"{ddm_std:.3f}", '10k error samples'])

    # VRAM usage
    if device.type == 'cuda':
        vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logging.info(f"\n  Peak VRAM Usage: {vram:.2f}MB")

    logging.info("\n✅ Hoàn thành Complexity Experiment (H5).")


if __name__ == "__main__":
    run_complexity_experiment()
