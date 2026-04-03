import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import pyarrow.parquet as pq
from collections import Counter
import joblib
import warnings

warnings.filterwarnings('ignore')

# Import mô hình từ folder model
from model.model import Hybrid_TKAN

# ================= CẤU HÌNH TỐI ƯU GPU =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              

PHASE_NAME = "Phase1_GPU_Optimization"
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed", "Phase1_Train")
PLOT_DIR = os.path.join(PROJECT_ROOT, "reports", "plots", PHASE_NAME)
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "global_scaler.pkl")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 2048        
SEQ_LENGTH = 10
LEARNING_RATE = 1e-3     
EARLY_STOP_PATIENCE = 5  
NUM_WORKERS = 4          # Số lượng luồng nạp data từ RAM vào VRAM

# ================= CẤU HÌNH LOGGING XUẤT RA FILE =================
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{PHASE_NAME}_training_VRAM.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"🚀 THIẾT BỊ SỬ DỤNG: {device}")
logging.info(f"📝 Log đang được tự động ghi vào: {LOG_FILE}")

# ================= LỚP DATASET & UTILS =================
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length=10, label_mapping=None, feature_cols=None, scaler=None):
        X_df = df.drop(columns=['Label'], errors='ignore')
        
        if feature_cols is not None:
            missing_cols = set(feature_cols) - set(X_df.columns)
            for c in missing_cols: X_df[c] = 0.0
            X_df = X_df[feature_cols]
            
        if scaler is not None:
            self.features = scaler.transform(X_df.values).astype(np.float32)
        else:
            self.features = X_df.values.astype(np.float32)
            
        self.labels_text = df['Label'].values
        self.label_mapping = label_mapping
            
        self.y = np.array([self.label_mapping.get(lbl, 0) for lbl in self.labels_text], dtype=np.int64)
        self.seq_length = seq_length
        self.num_samples = len(self.features) - self.seq_length + 1
        
    def __len__(self): return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        window_x = self.features[idx : idx + self.seq_length]
        target_y = self.y[idx + self.seq_length - 1]
        return torch.tensor(window_x), torch.tensor(target_y)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best_loss = None; self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None: self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_loss = val_loss; self.counter = 0

def plot_confusion_matrix(y_true, y_pred, labels_text, title, filename):
    labels_idx = np.arange(len(labels_text))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_text, yticklabels=labels_text)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{filename}.png")); plt.close()

# ================= VÒNG LẶP HUẤN LUYỆN =================
def train():
    all_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
    if not all_files: return logging.error(f"❌ Không thấy file tại {DATA_DIR}")

    logging.info("⚖️ Đang tải Global Scaler...")
    global_scaler = joblib.load(SCALER_PATH)

    first_df = pq.read_table(all_files[0]).to_pandas()
    GLOBAL_FEATURE_COLS = first_df.drop(columns=['Label'], errors='ignore').columns.tolist()
    
    logging.info("🔍 Đang quét toàn bộ file để chốt danh sách Nhãn (Labels)...")
    temp_labels = set()
    for f in tqdm(all_files, desc="Quét Labels", dynamic_ncols=True):
        temp_labels.update(pq.read_table(f, columns=['Label']).to_pandas()['Label'].unique())
        
    GLOBAL_LABEL_MAP = {lbl: idx for idx, lbl in enumerate(sorted(list(temp_labels)))}
    inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
    class_names = [inv_label_map[i] for i in range(len(GLOBAL_LABEL_MAP))]
    logging.info(f"🎯 Phát hiện tổng cộng {len(class_names)} nhãn.")

    logging.info(f"🧠 ĐANG NẠP TOÀN BỘ FILE LÊN 512GB RAM VÀ CHIA 80/20 THEO TỪNG NHÃN...")
    train_datasets_list = []
    val_datasets_list = []
    global_class_counts = {i: 0 for i in range(len(GLOBAL_LABEL_MAP))}
    
    for file_path in tqdm(all_files, desc="Nạp RAM & Chia", dynamic_ncols=True):
        full_df = pq.read_table(file_path).to_pandas()
        train_list, val_list = [], []
        
        for label, group in full_df.groupby('Label'):
            mapped_label = GLOBAL_LABEL_MAP[label]
            global_class_counts[mapped_label] += len(group)
            
            split_idx = int(len(group) * 0.8)
            train_list.append(group.iloc[:split_idx])
            if split_idx < len(group):
                val_list.append(group.iloc[split_idx:])
                
        train_df = pd.concat(train_list).sort_index().reset_index(drop=True)
        val_df = pd.concat(val_list).sort_index().reset_index(drop=True)
        
        train_ds = TimeSeriesDataset(train_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        val_ds = TimeSeriesDataset(val_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        
        if len(train_ds) > 0: train_datasets_list.append(train_ds)
        if len(val_ds) > 0: val_datasets_list.append(val_ds)
            
        del full_df, train_list, val_list, train_df, val_df 

    logging.info("⚖️ Đang tính toán trọng số cân bằng lớp (Class Weights)...")
    total_samples = sum(global_class_counts.values())
    num_classes = len(GLOBAL_LABEL_MAP)
    class_weights = [total_samples / (num_classes * global_class_counts[i]) if global_class_counts[i] > 0 else 0 
                     for i in range(num_classes)]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Hợp nhất dữ liệu toàn cục
    global_train_ds = ConcatDataset(train_datasets_list)
    global_val_ds = ConcatDataset(val_datasets_list)
    
    # Shuffle tự nhiên, không dùng Sampler để tránh lỗi tràn 2^24
    train_loader = DataLoader(global_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(global_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = Hybrid_TKAN(input_features=len(GLOBAL_FEATURE_COLS), num_classes=len(GLOBAL_LABEL_MAP)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # Tích hợp trọng số phạt
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    epoch = 1
    while True:
        logging.info(f"\n{'='*20} EPOCH {epoch} {'='*20}")
        
        # --- PHASE 1: TRAINING ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_batches = 0
        t_preds, t_labels = [], []
        lp, lr, lf1 = 0.0, 0.0, 0.0 

        pbar_train = tqdm(train_loader, desc=f"🚂 [TRAIN] Ep {epoch}", dynamic_ncols=True)
        for x, y in pbar_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item(); train_batches += 1
            _, pred = outputs.max(1)
            train_total += y.size(0); train_correct += pred.eq(y).sum().item()
            
            t_preds.extend(pred.cpu().numpy()); t_labels.extend(y.cpu().numpy())

            if train_batches % 100 == 0:
                p, r, f1, _ = precision_recall_fscore_support(t_labels, t_preds, average='macro', zero_division=0)
                lp, lr, lf1 = p, r, f1

            pbar_train.set_postfix({'L': f"{train_loss/train_batches:.3f}", 'A': f"{100.*train_correct/train_total:.1f}%", 
                                   'P': f"{lp:.2f}", 'R': f"{lr:.2f}", 'F1': f"{lf1:.2f}"})

        # --- PHASE 2: VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_batches = 0
        v_preds, v_labels = [], []
        vp, vr, vf1 = 0.0, 0.0, 0.0

        pbar_val = tqdm(val_loader, desc=f"🔍 [VALID] Ep {epoch}", dynamic_ncols=True)
        with torch.no_grad():
            for x, y in pbar_val:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item(); val_batches += 1
                _, pred = outputs.max(1)
                val_total += y.size(0); val_correct += pred.eq(y).sum().item()
                v_preds.extend(pred.cpu().numpy()); v_labels.extend(y.cpu().numpy())

                if val_batches % 50 == 0:
                    p, r, f1, _ = precision_recall_fscore_support(v_labels, v_preds, average='macro', zero_division=0)
                    vp, vr, vf1 = p, r, f1

                pbar_val.set_postfix({'L': f"{val_loss/val_batches:.3f}", 'A': f"{100.*val_correct/val_total:.1f}%",
                                     'P': f"{vp:.2f}", 'R': f"{vr:.2f}", 'F1': f"{vf1:.2f}"})

        # --- TỔNG KẾT EPOCH ---
        p, r, f1, _ = precision_recall_fscore_support(v_labels, v_preds, average='macro', zero_division=0)
        logging.info(f"📊 EPOCH {epoch} TỔNG KẾT: Val_Acc: {100.*val_correct/val_total:.2f}% | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

        plot_confusion_matrix(v_labels, v_preds, class_names, f"Đa lớp - Ep {epoch}", f"cm_multi_epoch_{epoch}")
        bin_labels = [0 if 'benign' in inv_label_map[i].lower() else 1 for i in v_labels]
        bin_preds = [0 if 'benign' in inv_label_map[i].lower() else 1 for i in v_preds]
        plot_confusion_matrix(bin_labels, bin_preds, ['Bình thường', 'Tấn công'], f"Nhị phân - Ep {epoch}", f"cm_bin_epoch_{epoch}")

        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"tkan_vram_epoch_{epoch}.pth"))
        
        early_stopping(val_loss/val_batches)
        if early_stopping.early_stop: 
            logging.info(f"🛑 Kích hoạt Early Stopping tại Epoch {epoch}! Mô hình đã hội tụ.")
            break
        epoch += 1

if __name__ == "__main__":
    train()