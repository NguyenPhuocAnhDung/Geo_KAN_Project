import os
import sys
import logging
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from river import drift
except ImportError:
    print("❌ Vui lòng cài đặt river: pip install river")
    sys.exit(1)

from src.data_preprocess.continual_loader import load_and_merge_data
from src.model.model import HierarchicalDriftTKAN
from src.training.train_stream import evaluate_chunk, finetune_on_chunk, StreamTimeSeriesDataset

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints', 'ContinualStream_DriftTKAN')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports', 'Baselines')
os.makedirs(REPORTS_DIR, exist_ok=True)
METRICS_CSV = os.path.join(REPORTS_DIR, 'baseline_metrics.csv')

SEQ_LENGTH = 10
BATCH_SIZE = 512
NUM_WORKERS = 4
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BASELINES] - %(message)s')

def run_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Khởi chạy Baseline Comparision trên {device}")
    
    from src.training.train_stream import GLOBAL_FEATURE_COLS, GLOBAL_LABEL_MAP
    
    train_df, val_df, test_df_list, test_df_names, all_labels_set, global_scaler = load_and_merge_data(
        use_cache=True, 
        phase1_ratio=0.1
    )
    
    inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
    class_names = [inv_label_map[i] for i in range(len(GLOBAL_LABEL_MAP))]
    NUM_CLASSES = len(class_names)
    
    model = HierarchicalDriftTKAN(
        input_features=len(GLOBAL_FEATURE_COLS),
        num_classes=NUM_CLASSES
    ).to(device)
    
    phase1_ckpt = os.path.join(CHECKPOINT_DIR, "phase1_baseline.pth")
    if os.path.exists(phase1_ckpt):
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        logging.info("✅ Đã load mô hình Phase 1.")
    else:
        logging.error("❌ Không tìm thấy mô hình Phase 1. Vui lòng chạy train_stream.py trước.")
        return
        
    criterion = nn.CrossEntropyLoss()
    
    detectors = {
        'ADWIN': drift.ADWIN(),
        'PageHinkley': drift.PageHinkley(),
        'DDM': drift.DDM(),
        'EDDM': drift.EDDM()
    }
    
    with open(METRICS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Chunk', 'F1_Macro', 'Detector', 'Drift_Detected', 'Total_Drifts'])
    
    step_counter = 0
    drift_counts = {name: 0 for name in detectors.keys()}
    
    for i, (chunk_df, chunk_name) in enumerate(zip(test_df_list, test_df_names)):
        step_counter += 1
        logging.info(f"🔄 ================= Xử lý Chunk {chunk_name} =================")
        
        chunk_ds = StreamTimeSeriesDataset(chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in chunk_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        f1_chunk = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        logging.info(f"📊 F1-macro trên {chunk_name}: {f1_chunk:.4f}")
        
        error_stream = [0 if p == l else 1 for p, l in zip(all_preds, all_labels)]
        detected_status = {name: False for name in detectors.keys()}
        
        for name, detector in detectors.items():
            for error in error_stream:
                detector.update(error)
                if detector.drift_detected:
                    detected_status[name] = True
                    drift_counts[name] += 1
                    break
                    
            status_str = "🔴 Phát hiện" if detected_status[name] else "🟢 Ổn định"
            logging.info(f"[{name}] {status_str} (Tổng: {drift_counts[name]})")
            
            with open(METRICS_CSV, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([step_counter, chunk_name, f"{f1_chunk:.4f}", name, str(detected_status[name]), drift_counts[name]])
        
        any_drift = any(detected_status.values())
        if any_drift:
            logging.info(f"⚡ Có baseline phát hiện Drift. Tiến hành Fine-tune trên Chunk hiện tại...")
            model.set_adaptation_mode(is_drifting=True)
            finetune_on_chunk(model, chunk_loader, criterion, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR)
            model.set_adaptation_mode(is_drifting=False)
            
    logging.info("✅ Hoàn thành Experiment B!")

if __name__ == "__main__":
    run_baselines()
