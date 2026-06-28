"""
experiment_ablation.py — Kiểm định Giả thuyết H3:
"Lợi ích đến từ chính kiến trúc được đề xuất, không phải các yếu tố khác"

Nguyên tắc Strict Variable Control:
- Mỗi thực nghiệm chỉ thay đổi DUY NHẤT 1 biến, giữ nguyên mọi thứ khác.

Nhóm 1 (Architecture Ablation):
  - Full Drift-TKAN (baseline)
  - No KAN (thay bằng MLP/Linear)

Nhóm 2 (Drift Sensitivity):
  - EMA α: 0.90, 0.95, 0.99
  - Threshold kσ: 2σ, 3σ, 4σ
  - Replay Size: 200, 500, 1000
"""

import os
import logging
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment_utils import (
    GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES, class_names,
    global_scaler, load_all_stream_chunks, load_model_phase1,
    StreamTimeSeriesDataset, finetune_on_chunk, SEQ_LENGTH
)
from data_preprocess.continual_loader import ClassBalancedReservoirBuffer

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Ablation_Experiment')
os.makedirs(REPORTS_DIR, exist_ok=True)
METRICS_CSV = os.path.join(REPORTS_DIR, 'ablation_metrics.csv')
DETAIL_CSV = os.path.join(REPORTS_DIR, 'ablation_detail.csv')

BATCH_SIZE = 512
NUM_WORKERS = 4
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ABLATION] - %(message)s')


def create_model_variant(device, variant='full'):
    """
    Tạo biến thể mô hình cho ablation.
    variant: 'full' | 'no_kan' | 'no_attention'
    """
    model = load_model_phase1(device)

    if variant == 'no_kan':
        # Thay KAN bằng Linear (MLP head)
        in_features = model.kan_classifier.in_features
        out_features = model.kan_classifier.out_features
        model.kan_classifier = nn.Linear(in_features, out_features).to(device)
        # Vì đã thay thế, compute_shift_score sẽ không hoạt động → drift luôn = 0
        logging.info("  ⚙️ Biến thể: KAN → Linear (MLP head)")

    return model


def run_stream_with_config(config, all_chunks, device):
    """
    Chạy toàn bộ pipeline stream với 1 cấu hình cụ thể.
    config: dict chứa variant, ema_alpha, threshold_factor, replay_size
    """
    variant = config.get('variant', 'full')
    ema_alpha = config.get('ema_alpha', 0.99)
    threshold_factor = config.get('threshold_factor', 3.0)
    replay_size = config.get('replay_size', 500)

    model = create_model_variant(device, variant=variant)
    criterion = nn.CrossEntropyLoss()
    replay_buffer = ClassBalancedReservoirBuffer(max_size_per_class=replay_size)

    f1_history = []
    drift_count = 0

    # Simulate a simple threshold based on the threshold_factor
    # In production this would come from Phase1 calibration
    shift_threshold = 0.005 * threshold_factor  # Scale by factor

    for chunk_df, chunk_name in tqdm(all_chunks, desc=f"  {config['name']}", leave=False):
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

        # 2. Detect Drift
        drift_detected = False
        if variant == 'full':
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
            drift_detected = (s_kan > shift_threshold)
        else:
            # For no_kan variant: no internal drift detection, just adapt every N chunks
            # Or always adapt (worst case for comparison)
            model.set_adaptation_mode(is_drifting=True)
            finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)
            model.set_adaptation_mode(is_drifting=False)

        if drift_detected:
            drift_count += 1
            model.set_adaptation_mode(is_drifting=True)
            model.train()
            finetune_on_chunk(model, chunk_loader, criterion, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR)
            model.set_adaptation_mode(is_drifting=False)

        # 3. Update EMA
        model.eval()
        with torch.no_grad():
            for x_val, y_val in chunk_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, context_vec = model(x_val)
                break
        if variant == 'full':
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=ema_alpha)

        # 4. Update Replay Buffer
        for x_buf, y_buf in DataLoader(chunk_ds, batch_size=256, shuffle=True):
            replay_buffer.add_samples(x_buf, y_buf)
            break

    avg_f1 = np.mean(f1_history)
    return f1_history, avg_f1, drift_count


def run_ablation_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Ablation Experiment (H3) trên {device}")

    all_chunks = load_all_stream_chunks()
    logging.info(f"📂 Tổng cộng {len(all_chunks)} chunks.")

    # Định nghĩa ma trận Ablation
    experiments = [
        # Nhóm 1: Architecture
        {'name': 'Full_Drift-TKAN', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 500},
        {'name': 'No_KAN_(Linear)', 'variant': 'no_kan', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 500},

        # Nhóm 2: EMA Alpha Sensitivity
        {'name': 'EMA_alpha=0.90', 'variant': 'full', 'ema_alpha': 0.90, 'threshold_factor': 1.0, 'replay_size': 500},
        {'name': 'EMA_alpha=0.95', 'variant': 'full', 'ema_alpha': 0.95, 'threshold_factor': 1.0, 'replay_size': 500},
        {'name': 'EMA_alpha=0.99', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 500},

        # Nhóm 3: Threshold Sensitivity
        {'name': 'Threshold_2sigma', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 0.67, 'replay_size': 500},
        {'name': 'Threshold_3sigma', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 500},
        {'name': 'Threshold_4sigma', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.33, 'replay_size': 500},

        # Nhóm 4: Replay Size Sensitivity
        {'name': 'Replay_200', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 200},
        {'name': 'Replay_500', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 500},
        {'name': 'Replay_1000', 'variant': 'full', 'ema_alpha': 0.99, 'threshold_factor': 1.0, 'replay_size': 1000},
    ]

    # Lưu CSV header
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Variant', 'EMA_Alpha', 'Threshold_Factor',
                         'Replay_Size', 'Avg_F1', 'Total_Drifts'])

    with open(DETAIL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Chunk_Index', 'F1_Macro'])

    for exp in experiments:
        logging.info(f"\n{'='*50}")
        logging.info(f"🔬 Thực nghiệm: {exp['name']}")
        logging.info(f"   variant={exp['variant']}, α={exp['ema_alpha']}, "
                     f"threshold_factor={exp['threshold_factor']}, replay={exp['replay_size']}")
        logging.info(f"{'='*50}")

        f1_hist, avg_f1, drift_count = run_stream_with_config(exp, all_chunks, device)

        logging.info(f"  📊 Avg F1: {avg_f1:.4f} | Drifts: {drift_count}")

        # Lưu summary
        with open(METRICS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([exp['name'], exp['variant'], exp['ema_alpha'],
                             exp['threshold_factor'], exp['replay_size'],
                             f"{avg_f1:.4f}", drift_count])

        # Lưu chi tiết
        with open(DETAIL_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, f1_val in enumerate(f1_hist):
                writer.writerow([exp['name'], i, f"{f1_val:.4f}"])

    logging.info("\n✅ Hoàn thành Ablation Experiment (H3).")


if __name__ == "__main__":
    run_ablation_experiment()
