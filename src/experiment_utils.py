"""
experiment_utils.py — Module chia sẻ cho tất cả Experiment Scripts.

Cung cấp:
- GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP, NUM_CLASSES, class_names
- global_scaler
- load_phase_chunks(): trả về danh sách (chunk_df, chunk_name) cho từng Phase
- load_model_phase1(): load mô hình đã train Phase 1
- StreamTimeSeriesDataset: re-export từ train_stream
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import joblib
import pyarrow.parquet as pq

# ================= ĐƯỜNG DẪN =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "global_scaler.pkl")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", "ContinualStream_DriftTKAN_v2")

# ================= LOAD SCALER =================
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Global scaler không tồn tại tại {SCALER_PATH}. "
        "Vui lòng chạy build_global_scaler.py hoặc train_stream.py Phase 1 trước."
    )

global_scaler = joblib.load(SCALER_PATH)
GLOBAL_FEATURE_COLS = global_scaler.feature_names_in_.tolist()

# ================= QUÉT NHÃN TOÀN CỤC =================
def _scan_all_labels():
    """Quét toàn bộ Phase1/2/3 để tạo label map thống nhất."""
    all_labels = set()
    for phase in ["Phase1_Train", "Phase2_DriftTest", "Phase3_DomainShift"]:
        phase_dir = os.path.join(PROCESSED_DIR, phase)
        files = sorted(glob.glob(os.path.join(phase_dir, "**/*.parquet"), recursive=True))
        for f in files:
            try:
                labels = pq.read_table(f, columns=['Label']).to_pandas()['Label'].unique()
                all_labels.update(labels)
            except Exception:
                continue
    return {lbl: idx for idx, lbl in enumerate(sorted(list(all_labels)))}

GLOBAL_LABEL_MAP = _scan_all_labels()
NUM_CLASSES = len(GLOBAL_LABEL_MAP)
inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
class_names = [inv_label_map[i] for i in range(NUM_CLASSES)]

# ================= HÀM TẢI DỮ LIỆU =================
SAMPLE_FRACTION = 0.1
SEQ_LENGTH = 10

def load_phase_files(phase_name):
    """Tải danh sách file parquet từ một Phase."""
    phase_dir = os.path.join(PROCESSED_DIR, phase_name)
    return sorted(glob.glob(os.path.join(phase_dir, "**/*.parquet"), recursive=True))

def sample_dataframe(file_path, fraction=SAMPLE_FRACTION):
    """Đọc parquet và lấy mẫu ngẫu nhiên theo tỷ lệ (Stratified Sampling)."""
    df = pq.read_table(file_path).to_pandas()
    if fraction >= 1.0:
        return df
    sampled_groups = []
    for label, group in df.groupby('Label'):
        n_sample = max(1, int(len(group) * fraction))
        sampled_groups.append(group.sample(n=n_sample, random_state=42))
    return pd.concat(sampled_groups).reset_index(drop=True)

def load_phase_chunks(phase_name, fraction=SAMPLE_FRACTION):
    """
    Trả về danh sách các tuple (chunk_df, chunk_name) cho một Phase.
    Đây là cách chuẩn để đọc dữ liệu stream, giống với train_stream.py.
    """
    files = load_phase_files(phase_name)
    chunks = []
    for f in files:
        try:
            chunk_df = sample_dataframe(f, fraction=fraction)
            chunk_name = os.path.basename(f).replace('.parquet', '')
            if len(chunk_df) >= SEQ_LENGTH + 1:
                chunks.append((chunk_df, chunk_name))
        except Exception:
            continue
    return chunks

def load_all_stream_chunks(fraction=SAMPLE_FRACTION):
    """Tải toàn bộ chunks từ Phase2 và Phase3 (dùng cho experiment scripts)."""
    chunks = []
    chunks.extend(load_phase_chunks("Phase2_DriftTest", fraction))
    chunks.extend(load_phase_chunks("Phase3_DomainShift", fraction))
    return chunks

def get_phase_boundaries(fraction=SAMPLE_FRACTION):
    """
    Trả về dict chứa index ranh giới giữa các dataset.
    Dùng làm ground-truth drift onset cho Detection Delay.
    """
    boundaries = {}
    idx = 0
    for phase in ["Phase2_DriftTest", "Phase3_DomainShift"]:
        phase_chunks = load_phase_chunks(phase, fraction)
        if phase_chunks:
            boundaries[phase] = idx  # chunk index đầu tiên của phase mới
        idx += len(phase_chunks)
    return boundaries

# ================= LOAD MÔ HÌNH =================
from model.model import HierarchicalDriftTKAN

def load_model_phase1(device=None):
    """Load mô hình HierarchicalDriftTKAN đã train Phase 1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HierarchicalDriftTKAN(
        input_features=len(GLOBAL_FEATURE_COLS),
        num_classes=NUM_CLASSES
    ).to(device)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "phase1_baseline.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint Phase 1 không tồn tại tại {ckpt_path}. "
            "Vui lòng chạy train_stream.py trước."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

# ================= RE-EXPORT =================
from train_stream import StreamTimeSeriesDataset, FocalLoss, finetune_on_chunk, evaluate_chunk
