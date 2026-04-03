# import os
# import glob
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# import numpy as np
# import pandas as pd
# import logging
# from tqdm import tqdm
# import pyarrow.parquet as pq
# from collections import Counter
# import joblib
# import warnings

# warnings.filterwarnings('ignore')

# # Import mô hình từ folder model
# from model.model import Hybrid_TKAN

# # ================= CẤU HÌNH TỐI ƯU CPU & RAM =================
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              

# PHASE_NAME = "Phase1_CPU_RAM_Only"
# DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed", "Phase1_Train")
# PLOT_DIR = os.path.join(PROJECT_ROOT, "reports", "plots", PHASE_NAME)
# MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
# SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "global_scaler.pkl")

# os.makedirs(PLOT_DIR, exist_ok=True)
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# # Ép cứng sử dụng CPU (Phù hợp với server 64 cores)
# device = torch.device("cpu")

# # Hyperparameters
# BATCH_SIZE = 2048        
# SEQ_LENGTH = 10
# LEARNING_RATE = 1e-3     
# EARLY_STOP_PATIENCE = 5  
# NUM_WORKERS = 32         # Tận dụng tối đa sức mạnh đa nhân

# # ================= LOGGING =================
# LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_FILE = os.path.join(LOG_DIR, f"{PHASE_NAME}_training_ram.log")

# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# # ================= LỚP DATASET (ĐÃ CẬP NHẬT REINDEX) =================
# class TimeSeriesDataset(Dataset):
#     def __init__(self, df, seq_length=10, label_mapping=None, feature_cols=None, scaler=None):
#         # Tách X và y
#         X_df = df.drop(columns=['Label'], errors='ignore')
        
#         # CĂN CHỈNH CỘT: Đảm bảo X_df có đúng các cột mà Scaler yêu cầu (thiếu điền 0)
#         if feature_cols is not None:
#             X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)
            
#         # Chuẩn hóa dữ liệu
#         if scaler is not None:
#             self.features = scaler.transform(X_df.values).astype(np.float32)
#         else:
#             self.features = X_df.values.astype(np.float32)
            
#         # Xử lý nhãn
#         self.y = np.array([label_mapping.get(lbl, 0) for lbl in df['Label'].values], dtype=np.int64)
#         self.seq_length = seq_length
#         self.num_samples = len(self.features) - self.seq_length + 1
        
#     def __len__(self): return max(0, self.num_samples)
    
#     def __getitem__(self, idx):
#         window_x = self.features[idx : idx + self.seq_length]
#         target_y = self.y[idx + self.seq_length - 1]
#         return torch.tensor(window_x), torch.tensor(target_y)

# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0):
#         self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best_loss = None; self.early_stop = False
#     def __call__(self, val_loss):
#         if self.best_loss is None: self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience: self.early_stop = True
#         else: self.best_loss = val_loss; self.counter = 0

# def plot_confusion_matrix(y_true, y_pred, labels_text, title, filename):
#     labels_idx = np.arange(len(labels_text))
#     cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_text, yticklabels=labels_text)
#     plt.title(f'Confusion Matrix - {title}')
#     plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
#     plt.xticks(rotation=45, ha='right'); plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, f"{filename}.png")); plt.close()

# # ================= VÒNG LẶP HUẤN LUYỆN =================
# def train():
#     all_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
#     if not all_files: return logging.error(f"❌ Không thấy file tại {DATA_DIR}")

#     logging.info(f"🚀 THIẾT BỊ SỬ DỤNG: {device.type.upper()}")
    
#     # 1. TẢI SCALER VÀ LẤY DANH SÁCH CỘT CHUẨN
#     logging.info("⚖️ Đang tải Global Scaler và danh sách đặc trưng chuẩn...")
#     global_scaler = joblib.load(SCALER_PATH)
#     GLOBAL_FEATURE_COLS = global_scaler.feature_names_in_.tolist() # Lấy 84 cột từ Scaler
#     logging.info(f"✅ Hệ thống sẽ sử dụng {len(GLOBAL_FEATURE_COLS)} đặc trưng làm đầu vào.")

#     # 2. QUÉT NHÃN TOÀN CỤC
#     logging.info("🔍 Đang quét toàn bộ dữ liệu để chốt danh sách Nhãn...")
#     temp_labels = set()
#     for f in tqdm(all_files, desc="Quét Labels", dynamic_ncols=True):
#         temp_labels.update(pq.read_table(f, columns=['Label']).to_pandas()['Label'].unique())
        
#     GLOBAL_LABEL_MAP = {lbl: idx for idx, lbl in enumerate(sorted(list(temp_labels)))}
#     inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
#     class_names = [inv_label_map[i] for i in range(len(GLOBAL_LABEL_MAP))]
#     logging.info(f"🎯 Phát hiện tổng cộng {len(class_names)} nhãn: {class_names}")

#     # 3. NẠP DỮ LIỆU LÊN RAM VÀ CHIA 80/20
#     logging.info(f"🧠 ĐANG NẠP DỮ LIỆU LÊN 512GB RAM...")
#     train_datasets_list, val_datasets_list = [], []
#     global_class_counts = {i: 0 for i in range(len(GLOBAL_LABEL_MAP))}
    
#     for file_path in tqdm(all_files, desc="Nạp RAM", dynamic_ncols=True):
#         full_df = pq.read_table(file_path).to_pandas()
#         t_list, v_list = [], []
        
#         for label, group in full_df.groupby('Label'):
#             mapped_label = GLOBAL_LABEL_MAP[label]
#             global_class_counts[mapped_label] += len(group)
#             split_idx = int(len(group) * 0.8)
#             t_list.append(group.iloc[:split_idx])
#             if split_idx < len(group): v_list.append(group.iloc[split_idx:])
                
#         train_df = pd.concat(t_list).sort_index().reset_index(drop=True)
#         val_df = pd.concat(v_list).sort_index().reset_index(drop=True)
        
#         train_ds = TimeSeriesDataset(train_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
#         val_ds = TimeSeriesDataset(val_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        
#         if len(train_ds) > 0: train_datasets_list.append(train_ds)
#         if len(val_ds) > 0: val_datasets_list.append(val_ds)
#         del full_df, t_list, v_list, train_df, val_df 

#     # 4. TÍNH TOÁN TRỌNG SỐ PHẠT (CLASS WEIGHTS)
#     total_samples = sum(global_class_counts.values())
#     num_classes = len(GLOBAL_LABEL_MAP)
#     class_weights = [total_samples / (num_classes * global_class_counts[i]) if global_class_counts[i] > 0 else 0 
#                      for i in range(num_classes)]
#     class_weights_tensor = torch.FloatTensor(class_weights).to(device)
#     logging.info("⚖️ Đã tính toán xong Trọng số cân bằng lớp.")

#     # 5. KHỞI TẠO DATALOADER
#     train_loader = DataLoader(ConcatDataset(train_datasets_list), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
#     val_loader = DataLoader(ConcatDataset(val_datasets_list), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#     # 6. KHỞI TẠO MÔ HÌNH VỚI SỐ CỘT CHUẨN
#     model = Hybrid_TKAN(input_features=len(GLOBAL_FEATURE_COLS), num_classes=len(GLOBAL_LABEL_MAP)).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#     early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
#     epoch = 1
#     while True:
#         logging.info(f"\n{'='*20} EPOCH {epoch} {'='*20}")
#         model.train()
#         train_loss, train_correct, train_total = 0, 0, 0
#         t_preds, t_labels = [], []

#         pbar = tqdm(train_loader, desc=f"🚂 [TRAIN] Ep {epoch}", dynamic_ncols=True)
#         for x, y in pbar:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(x)
#             loss = criterion(outputs, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             train_loss += loss.item()
#             _, pred = outputs.max(1)
#             train_total += y.size(0); train_correct += pred.eq(y).sum().item()
#             t_preds.extend(pred.numpy()); t_labels.extend(y.numpy())

#             pbar.set_postfix({'Loss': f"{train_loss/(pbar.n+1):.3f}", 'Acc': f"{100.*train_correct/train_total:.1f}%"})

#         # VALIDATION
#         model.eval()
#         val_loss, val_correct, val_total = 0, 0, 0
#         v_preds, v_labels = [], []
#         with torch.no_grad():
#             for x, y in tqdm(val_loader, desc=f"🔍 [VALID] Ep {epoch}", dynamic_ncols=True):
#                 x, y = x.to(device), y.to(device)
#                 outputs = model(x)
#                 loss = criterion(outputs, y)
#                 val_loss += loss.item()
#                 _, pred = outputs.max(1)
#                 val_total += y.size(0); val_correct += pred.eq(y).sum().item()
#                 v_preds.extend(pred.numpy()); v_labels.extend(y.numpy())

#         p, r, f1, _ = precision_recall_fscore_support(v_labels, v_preds, average='macro', zero_division=0)
#         logging.info(f"📊 TỔNG KẾT EP {epoch}: Val_Acc: {100.*val_correct/val_total:.2f}% | F1: {f1:.4f}")

#         plot_confusion_matrix(v_labels, v_preds, class_names, f"Ep {epoch}", f"cm_epoch_{epoch}")
#         torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"tkan_cpu_epoch_{epoch}.pth"))
        
#         early_stopping(val_loss/len(val_loader))
#         if early_stopping.early_stop: break
#         epoch += 1

# if __name__ == "__main__":
#     train()


import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report # <--- Bổ sung report chi tiết
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import pyarrow.parquet as pq
from collections import Counter
import joblib
import warnings

warnings.filterwarnings('ignore')

from model.model import Hybrid_TKAN

# ================= CẤU HÌNH TỐI ƯU =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              

PHASE_NAME = "Phase1_CPU_RAM_Only"
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed", "Phase1_Train")
PLOT_DIR = os.path.join(PROJECT_ROOT, "reports", "plots", PHASE_NAME)
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "global_scaler.pkl")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cpu")

# Hyperparameters
BATCH_SIZE = 2048        
SEQ_LENGTH = 10
LEARNING_RATE = 1e-3     
EARLY_STOP_PATIENCE = 7  # <--- Tăng kiên nhẫn để đợi Scheduler hoạt động
NUM_WORKERS = 32         

# ================= LOGGING =================
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{PHASE_NAME}_training_ram.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length=10, label_mapping=None, feature_cols=None, scaler=None):
        X_df = df.drop(columns=['Label'], errors='ignore')
        
        if feature_cols is not None:
            X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)
            
        if scaler is not None:
            self.features = scaler.transform(X_df.values).astype(np.float32)
        else:
            self.features = X_df.values.astype(np.float32)
            
        self.y = np.array([label_mapping.get(lbl, 0) for lbl in df['Label'].values], dtype=np.int64)
        self.seq_length = seq_length
        self.num_samples = len(self.features) - self.seq_length + 1
        
    def __len__(self): return max(0, self.num_samples)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx : idx + self.seq_length]), torch.tensor(self.y[idx + self.seq_length - 1])

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
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
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_text, yticklabels=labels_text)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{filename}.png")); plt.close()

def train():
    all_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
    if not all_files: return logging.error("❌ Không thấy file dữ liệu!")

    logging.info(f"🚀 THIẾT BỊ SỬ DỤNG: CPU")
    global_scaler = joblib.load(SCALER_PATH)
    GLOBAL_FEATURE_COLS = global_scaler.feature_names_in_.tolist()
    
    logging.info("🔍 Đang quét toàn bộ dữ liệu để chốt danh sách Nhãn...")
    temp_labels = set()
    for f in tqdm(all_files, desc="Quét Labels", dynamic_ncols=True):
        temp_labels.update(pq.read_table(f, columns=['Label']).to_pandas()['Label'].unique())
        
    GLOBAL_LABEL_MAP = {lbl: idx for idx, lbl in enumerate(sorted(list(temp_labels)))}
    inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
    class_names = [inv_label_map[i] for i in range(len(GLOBAL_LABEL_MAP))]
    logging.info(f"🎯 Phát hiện {len(class_names)} nhãn: {class_names}")

    logging.info(f"🧠 ĐANG NẠP DỮ LIỆU LÊN 512GB RAM...")
    train_datasets_list, val_datasets_list = [], []
    global_class_counts = {i: 0 for i in range(len(GLOBAL_LABEL_MAP))}
    
    for file_path in tqdm(all_files, desc="Nạp RAM", dynamic_ncols=True):
        full_df = pq.read_table(file_path).to_pandas()
        t_list, v_list = [], []
        for label, group in full_df.groupby('Label'):
            mapped_label = GLOBAL_LABEL_MAP[label]
            global_class_counts[mapped_label] += len(group)
            split_idx = int(len(group) * 0.8)
            t_list.append(group.iloc[:split_idx])
            if split_idx < len(group): v_list.append(group.iloc[split_idx:])
                
        train_ds = TimeSeriesDataset(pd.concat(t_list), SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        val_ds = TimeSeriesDataset(pd.concat(v_list), SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
        if len(train_ds) > 0: train_datasets_list.append(train_ds)
        if len(val_ds) > 0: val_datasets_list.append(val_ds)

    # --- NÂNG CẤP: SMOOTHED CLASS WEIGHTS ---
    logging.info("⚖️ Đang tính toán Smoothed Class Weights (Chống bùng nổ Gradient)...")
    num_classes = len(GLOBAL_LABEL_MAP)
    # Dùng căn bậc 2 (sqrt) để làm mượt trọng số, giúp các nhãn nhỏ như Web_Attack học tốt hơn
    class_weights = [1.0 / np.sqrt(global_class_counts[i] + 1e-5) for i in range(num_classes)]
    weight_sum = sum(class_weights)
    class_weights = [w * num_classes / weight_sum for w in class_weights] # Chuẩn hóa
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    train_loader = DataLoader(ConcatDataset(train_datasets_list), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ConcatDataset(val_datasets_list), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = Hybrid_TKAN(input_features=len(GLOBAL_FEATURE_COLS), num_classes=num_classes).to(device)
    
    # --- NÂNG CẤP: OPTIMIZER & SCHEDULER CHỐNG OVERFITTING ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # L2 Regularization
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    
    epoch = 1
    while True:
        logging.info(f"\n{'='*20} EPOCH {epoch} {'='*20}")
        model.train()
        train_loss, train_batches = 0, 0

        pbar_train = tqdm(train_loader, desc=f"🚂 [TRAIN] Ep {epoch}", dynamic_ncols=True)
        for x, y in pbar_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar_train.set_postfix({'Loss': f"{train_loss/train_batches:.3f}"})

        model.eval()
        val_loss, val_batches = 0, 0
        v_preds, v_labels = [], []
        
        pbar_val = tqdm(val_loader, desc=f"🔍 [VALID] Ep {epoch}", dynamic_ncols=True)
        with torch.no_grad():
            for x, y in pbar_val:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_batches += 1
                _, pred = outputs.max(1)
                v_preds.extend(pred.cpu().numpy()); v_labels.extend(y.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        
        # --- NÂNG CẤP: BÁO CÁO ALL METRICS (CLASSIFICATION REPORT) ---
        report = classification_report(v_labels, v_preds, target_names=class_names, digits=4, zero_division=0)
        logging.info(f"\n📊 BÁO CÁO CHI TIẾT CÁC CHỈ SỐ EPOCH {epoch}:\n{report}")
        logging.info(f"👉 Validation Loss: {avg_val_loss:.4f}")

        plot_confusion_matrix(v_labels, v_preds, class_names, f"Ep {epoch}", f"cm_epoch_{epoch}")
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"tkan_cpu_epoch_{epoch}.pth"))
        
        # Cập nhật Learning Rate nếu Loss không giảm
        scheduler.step(avg_val_loss)
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop: 
            logging.info(f"🛑 Kích hoạt Early Stopping! Mô hình đã hội tụ ở Epoch {epoch}.")
            break
        epoch += 1

if __name__ == "__main__":
    train()