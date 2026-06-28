"""
==========================================================================
 train_stream.py - Prequential Evaluation (Test-Then-Train) trên RAM/CPU
 =========================================================================
 Script mô phỏng luồng dữ liệu liên tục (Continual Learning) qua các 
 giai đoạn thời gian: Phase1 (Train khởi tạo) → Phase2 (Drift Test) → Phase3 (Domain Shift).
 
 Quy trình tại mỗi Data Chunk:
   1. TEST TRƯỚC (Prequential): Dự đoán chunk mới, ghi Accuracy/F1.
   2. ĐO SHIFT SCORE: Tính Cosine Distance trên hệ số KAN (EMA).
   3. PHÁT HIỆN DRIFT: Nếu Shift Score > Ngưỡng → Kích hoạt Fine-tuning.
   4. TRAIN/ADAPT: Fine-tune (đóng băng BiLSTM) trên chunk mới + Replay Buffer.
   5. CẬP NHẬT BUFFER: Lưu mẫu mới vào ClassBalancedReplayBuffer.
   
 Tối ưu cho CPU / RAM (512GB).
==========================================================================
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import csv
from tqdm import tqdm
import pyarrow.parquet as pq
import joblib
import warnings
from collections import Counter
from sklearn.metrics import classification_report, f1_score
import argparse

warnings.filterwarnings('ignore')

# Khởi tạo Argparse cho Ablation Study
parser = argparse.ArgumentParser(description="Continual Stream Training với Ablation Study")
parser.add_argument("--disable_s_rep", action="store_true", help="Vô hiệu hóa Representation Drift (S_rep)")
parser.add_argument("--disable_s_kan", action="store_true", help="Vô hiệu hóa Decision Drift (S_KAN)")
parser.add_argument("--disable_entropy", action="store_true", help="Vô hiệu hóa Entropy (Predictive Uncertainty)")
args = parser.parse_args()

# Import các module nội bộ
from model.model import HierarchicalDriftTKAN
from data_preprocess.continual_loader import ClassBalancedReservoirBuffer, StreamDataset

# ================= 1. CẤU HÌNH ĐƯỜNG DẪN =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "global_scaler.pkl")

EXPERIMENT_NAME = "ContinualStream_DriftTKAN_v2"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", EXPERIMENT_NAME)
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", EXPERIMENT_NAME)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", EXPERIMENT_NAME)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= 2. HYPERPARAMETERS =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
SEQ_LENGTH = 10
FINETUNE_EPOCHS = 3          # Số epoch Fine-tune khi phát hiện Drift
FINETUNE_LR = 5e-4           # Learning rate thấp hơn cho Fine-tuning
DRIFT_THRESHOLD = 0.025       # Ngưỡng Cosine Distance để cắm cờ Drift (Auto-derived 3-Sigma)
EMA_ALPHA = 0.99             # Hệ số EMA cho tham chiếu KAN
REPLAY_BUFFER_SIZE = 500     # Số mẫu / class trong Replay Buffer
REPLAY_MIX_RATIO = 0.4       # 40% dữ liệu cũ, 60% dữ liệu mới
NUM_WORKERS = 8              # Giảm workers so với full train để tiết kiệm RAM
SAMPLE_FRACTION = 0.1        # 10% dữ liệu (11.5 triệu mẫu) là lý tưởng chống OOM
INITIAL_TRAIN_EPOCHS = 10    # Số epoch train khởi tạo (Phase 1)

# ================= 3. LOGGING =================
LOG_FILE = os.path.join(LOG_DIR, "stream_training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# File CSV lưu metrics theo thời gian (để vẽ đồ thị Money Shot)
METRICS_CSV = os.path.join(REPORT_DIR, "stream_metrics.csv")

# ================= 4. FOCAL LOSS (ỔN ĐỊNH TOÁN HỌC) =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Sử dụng log_softmax để chống nổ số (NaN)
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        ce_loss = F.nll_loss(log_pt, targets, weight=self.alpha, reduction='none')
        # Tính focal term
        pt_target = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - pt_target) ** self.gamma * ce_loss).mean()
        return focal_loss

# ================= 5. DATASET CHO STREAM =================
class StreamTimeSeriesDataset(Dataset):
    """Chuyển DataFrame thành chuỗi thời gian 3D (đã chuẩn hóa)."""
    def __init__(self, df, seq_length, label_mapping, feature_cols, scaler):
        X_df = df.drop(columns=['Label'], errors='ignore')
        X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)
        self.features = scaler.transform(X_df.values).astype(np.float32)
        # Chống NaN triệt để từ mảng NumPy
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.y = np.array([label_mapping.get(lbl, 0) for lbl in df['Label'].values], dtype=np.int64)
        self.seq_length = seq_length
        self.num_samples = len(self.features) - self.seq_length + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        window_x = self.features[idx: idx + self.seq_length]
        target_y = self.y[idx + self.seq_length - 1]
        return torch.tensor(window_x), torch.tensor(target_y)

# ================= 6. HÀM ĐÁNH GIÁ (PREQUENTIAL TEST) =================
def evaluate_chunk(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    total_entropy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # Nan padding
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            logits, context_vector = model(x)
            
            # Tính Entropy
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
            total_entropy += entropy.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    if len(all_targets) == 0:
        return 0.0, "Empty chunk", 0.0
    
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    report = classification_report(all_targets, all_preds, 
                                   target_names=class_names, 
                                   labels=np.arange(len(class_names)),
                                   digits=4, zero_division=0)
    avg_entropy = total_entropy / max(num_batches, 1)
    return f1_macro, report, avg_entropy

# ================= 7. HÀM FINE-TUNE (ADAPTATION) =================
def finetune_on_chunk(model, train_loader, criterion, epochs=3, lr=5e-4):
    """Fine-tune mô hình trên chunk mới (đã đóng băng BiLSTM)."""
    # Chỉ tối ưu các tham số đang requires_grad = True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    model.train()
    for ep in range(1, epochs + 1):
        total_loss, total_batches = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            optimizer.zero_grad()
            logits, context_vector = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        logging.info(f"      [Fine-tune] Epoch {ep}/{epochs} - Loss: {avg_loss:.4f}")

# ================= 8. CHUẨN BỊ CÁC PHASE DỮ LIỆU =================
def load_phase_files(phase_name):
    """Tải danh sách file parquet từ một Phase."""
    phase_dir = os.path.join(PROCESSED_DIR, phase_name)
    files = sorted(glob.glob(os.path.join(phase_dir, "**/*.parquet"), recursive=True))
    return files

def sample_dataframe(file_path, fraction=0.05):
    """Đọc parquet và lấy mẫu ngẫu nhiên theo tỷ lệ (Stratified Sampling)."""
    df = pq.read_table(file_path).to_pandas()
    if fraction >= 1.0:
        return df
    # Stratified sampling: giữ tỷ lệ nhãn
    sampled_groups = []
    for label, group in df.groupby('Label'):
        n_sample = max(1, int(len(group) * fraction))
        sampled_groups.append(group.sample(n=n_sample, random_state=42))
    return pd.concat(sampled_groups).reset_index(drop=True)

# ================= 9. VÒNG LẶP CHÍNH (MAIN STREAM LOOP) =================
def main():
    logging.info("🔥" * 20)
    logging.info("🔥 BẮT ĐẦU THỰC NGHIỆM CONTINUAL LEARNING - DRIFT DETECTION 🔥")
    logging.info("🔥" * 20)
    
    # --- A. Tải Scaler và xác định Global Features ---
    logging.info("⚙️ Đang tải Global Scaler...")
    global_scaler = joblib.load(SCALER_PATH)
    GLOBAL_FEATURE_COLS = global_scaler.feature_names_in_.tolist()
    logging.info(f"✅ Scaler có {len(GLOBAL_FEATURE_COLS)} đặc trưng.")
    
    # --- B. Quét nhãn toàn cục qua TẤT CẢ các phase ---
    logging.info("🔍 Đang quét nhãn toàn cục qua Phase1, Phase2, Phase3...")
    all_labels_set = set()
    for phase in ["Phase1_Train", "Phase2_DriftTest", "Phase3_DomainShift"]:
        phase_files = load_phase_files(phase)
        for f in phase_files:
            try:
                all_labels_set.update(pq.read_table(f, columns=['Label']).to_pandas()['Label'].unique())
            except:
                continue
    GLOBAL_LABEL_MAP = {lbl: idx for idx, lbl in enumerate(sorted(list(all_labels_set)))}
    inv_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}
    class_names = [inv_label_map[i] for i in range(len(GLOBAL_LABEL_MAP))]
    NUM_CLASSES = len(class_names)
    logging.info(f"🎯 Phát hiện {NUM_CLASSES} nhãn: {class_names}")
    
    # --- D. Khởi tạo mô hình ---
    logging.info("Khởi tạo mô hình HierarchicalDriftTKAN...")
    model = HierarchicalDriftTKAN(
        input_features=len(GLOBAL_FEATURE_COLS),
        num_classes=NUM_CLASSES,
    ).to(device)
    model.set_adaptation_mode(is_drifting=False)  # Mở khóa toàn bộ cho Phase 1
    logging.info("🧠 Đã khởi tạo mô hình HierarchicalDriftTKAN (Full Unfreeze).")
    
    # --- D. Khởi tạo Replay Buffer ---
    replay_buffer = ClassBalancedReservoirBuffer(max_size_per_class=REPLAY_BUFFER_SIZE)
    
    # --- E. Khởi tạo file CSV ghi metrics ---
    with open(METRICS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Phase', 'Chunk', 'F1_Macro', 'S_rep', 'S_KAN', 'Entropy', 'Drift_Type', 'Action'])
    
    step_counter = 0
    
    # ===================================================================
    #                    PHASE 1: HUẤN LUYỆN KHỞI TẠO
    # ===================================================================
    logging.info("\n" + "=" * 60)
    logging.info("🌟 PHASE 1: HUẤN LUYỆN KHỞI TẠO (Train trên Phase1_Train)")
    logging.info("=" * 60)
    
    phase1_files = load_phase_files("Phase1_Train")
    logging.info(f"📂 Tìm thấy {len(phase1_files)} file parquet trong Phase1_Train.")
    
    # Nạp toàn bộ Phase1 vào RAM (có sampling)
    phase1_dfs = []
    for f in tqdm(phase1_files, desc="📥 Nạp Phase1"):
        df = sample_dataframe(f, fraction=SAMPLE_FRACTION)
        phase1_dfs.append(df)
    
    phase1_all = pd.concat(phase1_dfs, ignore_index=True)
    del phase1_dfs
    logging.info(f"✅ Đã nạp Phase1: {len(phase1_all):,} mẫu (sau sampling {SAMPLE_FRACTION*100:.0f}%).")
    
    # Chia 80/20
    train_groups, val_groups = [], []
    for label, group in phase1_all.groupby('Label'):
        split_idx = int(len(group) * 0.8)
        train_groups.append(group.iloc[:split_idx])
        if split_idx < len(group):
            val_groups.append(group.iloc[split_idx:])
    
    train_df = pd.concat(train_groups).reset_index(drop=True)
    val_df = pd.concat(val_groups).reset_index(drop=True)
    del phase1_all, train_groups, val_groups
    
    train_ds = StreamTimeSeriesDataset(train_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
    val_ds = StreamTimeSeriesDataset(val_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Tính class weights
    label_counts = Counter(train_df['Label'].values)
    total_samples = sum(label_counts.values())
    class_weights = torch.zeros(NUM_CLASSES)
    for lbl, count in label_counts.items():
        idx = GLOBAL_LABEL_MAP.get(lbl, 0)
        class_weights[idx] = 1.0 / np.sqrt(count + 1e-5)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Vòng lặp Train khởi tạo
    for epoch in range(1, INITIAL_TRAIN_EPOCHS + 1):
        logging.info(f"\n--- Phase1 Epoch {epoch}/{INITIAL_TRAIN_EPOCHS} ---")
        model.train()
        total_loss, total_batches = 0, 0
        
        for x, y in tqdm(train_loader, desc=f"🚂 [Train] Ep {epoch}", dynamic_ncols=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        
        avg_train_loss = total_loss / max(total_batches, 1)
        
        # Validation
        f1_val, report, _ = evaluate_chunk(model, val_loader, class_names)
        logging.info(f"📊 Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val F1-macro={f1_val:.4f}")
        logging.info(f"\n{report}")
        
        scheduler.step(avg_train_loss)
    
    # Lưu checkpoint Phase 1
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "phase1_baseline.pth"))
    logging.info("💾 Đã lưu mô hình nền tảng Phase 1.")
    
    # Thiết lập mốc tham chiếu KAN và Class-wise Prototypes ban đầu
    model.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            _, context_vec = model(x_val)
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=0.0) # Khởi tạo mạnh
            
    # Tính Ngưỡng Động (Adaptive Thresholds) từ Phase 1
    logging.info("🔍 Đang phân tích nhiễu phân phối Phase 1 để trích xuất ngưỡng (P99, P95)...")
    shift_scores_rep_p1 = []
    shift_scores_kan_p1 = []
    entropy_p1 = []
    
    model.set_adaptation_mode(is_drifting=True)
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits, context_vec = model(x_val)
            
            # Tính Hierarchical Drift
            s_rep, s_kan = model.compute_hierarchical_drift(context_vec, logits)
            shift_scores_rep_p1.append(s_rep)
            shift_scores_kan_p1.append(s_kan)
            
            # Tính Entropy
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean().item()
            entropy_p1.append(entropy)
            
            # Cập nhật EMA Ground-truth
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)
            
    global THRESHOLD_REP, THRESHOLD_KAN, THRESHOLD_ENTROPY
    THRESHOLD_REP = max(np.percentile(shift_scores_rep_p1, 99), 0.005)
    THRESHOLD_KAN = max(np.percentile(shift_scores_kan_p1, 99), 0.005)
    THRESHOLD_ENTROPY = np.percentile(entropy_p1, 95)
    
    logging.info(f"🎯 Đã tự động kích hoạt ngưỡng: Rep P99={THRESHOLD_REP:.6f} | KAN P99={THRESHOLD_KAN:.6f} | Entropy P95={THRESHOLD_ENTROPY:.4f}")
    
    model.set_adaptation_mode(is_drifting=False)
    
    # Nạp một phần dữ liệu Phase1 vào Replay Buffer
    for x, y in DataLoader(train_ds, batch_size=512, shuffle=True):
        replay_buffer.add_samples(x, y)  # Lưu toàn bộ chuỗi thời gian (SEQ_LENGTH)
        if sum(len(v) for v in replay_buffer.buffer_y.values()) >= REPLAY_BUFFER_SIZE * NUM_CLASSES:
            break
    logging.info(f"📦 Replay Buffer đã nạp: {sum(len(v) for v in replay_buffer.buffer_y.values())} mẫu.")
    
    # Giải phóng RAM Phase 1
    del train_df, val_df, train_ds, val_ds, train_loader, val_loader
    
    step_counter += 1
    with open(METRICS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([step_counter, 'Phase1_Train', 'full_init', f"{f1_val:.4f}", '0.0', '0.0', '0.0', 'No Drift', 'initial_train'])
    
    # ===================================================================
    #          PHASE 2 & 3: PREQUENTIAL EVALUATION (TEST-THEN-TRAIN)
    # ===================================================================
    stream_phases = [
        ("Phase2_DriftTest", "Phase2_DriftTest"),
        ("Phase3_DomainShift", "Phase3_DomainShift")
    ]
    
    for phase_label, phase_dir_name in stream_phases:
        logging.info("\n" + "=" * 60)
        logging.info(f"🌊 BẮT ĐẦU LUỒNG: {phase_label}")
        logging.info("=" * 60)
        
        phase_files = load_phase_files(phase_dir_name)
        if not phase_files:
            logging.warning(f"⚠️ Không tìm thấy dữ liệu cho {phase_label}. Bỏ qua.")
            continue
        logging.info(f"📂 Tìm thấy {len(phase_files)} file parquet.")
        
        for file_idx, file_path in enumerate(phase_files):
            chunk_name = os.path.basename(file_path).replace('.parquet', '')
            step_counter += 1
            
            logging.info(f"\n--- [{phase_label}] Chunk {file_idx+1}/{len(phase_files)}: {chunk_name} ---")
            
            # Đọc và lấy mẫu chunk
            try:
                chunk_df = sample_dataframe(file_path, fraction=SAMPLE_FRACTION)
            except Exception as e:
                logging.error(f"❌ Lỗi đọc file {chunk_name}: {e}")
                continue
            
            if len(chunk_df) < SEQ_LENGTH + 1:
                logging.warning(f"⚠️ Chunk {chunk_name} quá nhỏ ({len(chunk_df)} dòng). Bỏ qua.")
                continue
            
            chunk_ds = StreamTimeSeriesDataset(chunk_df, SEQ_LENGTH, GLOBAL_LABEL_MAP, GLOBAL_FEATURE_COLS, global_scaler)
            chunk_loader = DataLoader(chunk_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            # ========== BƯỚC 1: TEST TRƯỚC (Prequential) ==========
            f1_chunk, report, avg_entropy = evaluate_chunk(model, chunk_loader, class_names)
            logging.info(f"📊 [TEST] F1-macro trên {chunk_name}: {f1_chunk:.4f} | Entropy: {avg_entropy:.4f}")
            logging.info(f"\n{report}")
            
            # ========== BƯỚC 2: ĐO SHIFT SCORE VÀ PHÁT HIỆN DRIFT ==========
            # Để đo được độ lệch, mô hình cần được "thích nghi nhẹ" với dữ liệu mới
            # Khóa BiLSTM, chỉ cho KAN và Attention thích nghi
            model.set_adaptation_mode(is_drifting=True)
            finetune_on_chunk(model, chunk_loader, criterion, epochs=1, lr=1e-4)
            
            # Lấy vector biểu diễn của Chunk sau khi thích nghi
            model.eval()
            with torch.no_grad():
                for x_val, _ in chunk_loader:
                    x_val = x_val.to(device)
                    logits, context_vec = model(x_val)
                    # Tính Hierarchical Drift
                    s_rep, s_kan = model.compute_hierarchical_drift(context_vec, logits)
                    break # Chỉ lấy 1 batch đại diện
                    
            # Đánh giá Drift Type (Có hỗ trợ Ablation Study)
            drift_type = "No Drift"
            drift_detected = False
            
            # Áp dụng Ablation Flags
            effective_s_rep = 0.0 if args.disable_s_rep else s_rep
            effective_s_kan = 0.0 if args.disable_s_kan else s_kan
            
            if effective_s_rep > THRESHOLD_REP and effective_s_kan <= THRESHOLD_KAN:
                drift_type = "Type I: Feature Shift"
                drift_detected = True
            elif effective_s_kan > THRESHOLD_KAN and effective_s_rep <= THRESHOLD_REP:
                drift_type = "Type II: Decision Shift"
                drift_detected = True
            elif effective_s_rep > THRESHOLD_REP and effective_s_kan > THRESHOLD_KAN:
                drift_type = "Type III: Severe Concept Drift"
                drift_detected = True
            
            status_emoji = "🔴 DRIFT DETECTED!" if drift_detected else "🟢 Ổn định"
            logging.info(f"📈 S_rep: {s_rep:.6f} (Ngưỡng: {THRESHOLD_REP:.6f}) {'[DISABLED]' if args.disable_s_rep else ''}")
            logging.info(f"📈 S_KAN: {s_kan:.6f} (Ngưỡng: {THRESHOLD_KAN:.6f}) {'[DISABLED]' if args.disable_s_kan else ''}")
            logging.info(f"🎯 Kết luận: {drift_type} → {status_emoji}")
            
            if drift_detected and avg_entropy > THRESHOLD_ENTROPY and not args.disable_entropy:
                logging.warning("⚠️ CẢNH BÁO: POTENTIAL NOVEL ATTACK DETECTED! (Confidence Entropy vượt ngưỡng P95)")
                
            # Mở lại chế độ train nếu cần cho Fine-tune
            model.train()
            
            # ========== BƯỚC 3: FINE-TUNE SÂU NẾU CÓ DRIFT ==========
            action = "no_action"
            if drift_detected:
                action = "fine_tune"
                logging.info(f"⚡ Kích hoạt chế độ THÍCH NGHI SÂU (Mix Buffer)...")
                
                # Lấy dữ liệu từ Replay Buffer
                replay_batch_size = int(BATCH_SIZE * REPLAY_MIX_RATIO)
                buf_x, buf_y = replay_buffer.get_balanced_batch(replay_batch_size)
                
                if buf_x is not None:
                    # Lấy an toàn 1 batch từ chunk mới bằng iterator
                    chunk_iter = iter(chunk_loader)
                    try:
                        x_stream_batch, y_stream_batch = next(chunk_iter)
                    except StopIteration:
                        x_stream_batch = torch.empty((0, SEQ_LENGTH, len(GLOBAL_FEATURE_COLS)), dtype=torch.float32)
                        y_stream_batch = torch.empty((0,), dtype=torch.long)
                        
                    # Tạo dataset mix: chunk mới + buffer cũ (buf_x đã là chuỗi 3D)
                    mix_ds = StreamDataset(
                        x_stream=x_stream_batch,
                        y_stream=y_stream_batch,
                        x_buffer=buf_x,
                        y_buffer=buf_y
                    )
                    mix_loader = DataLoader(mix_ds, batch_size=BATCH_SIZE, shuffle=True)
                else:
                    mix_loader = chunk_loader
                
                # Fine-tune sâu thêm
                finetune_on_chunk(model, mix_loader, criterion, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR)
                
                # Đánh giá lại sau Fine-tune
                f1_after, _, _ = evaluate_chunk(model, chunk_loader, class_names)
                logging.info(f"📊 [SAU FINE-TUNE] F1-macro: {f1_chunk:.4f} → {f1_after:.4f}")
                
            # Mở khóa lại toàn bộ cho an toàn
            model.set_adaptation_mode(is_drifting=False)
            
            # Lấy context vector cuối cùng của chunk để cập nhật EMA
            model.eval()
            with torch.no_grad():
                for x_val, y_val in chunk_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    _, context_vec = model(x_val)
                    break
            # Cập nhật EMA tham chiếu bằng trọng số hiện tại (Ground Truth Update)
            model.update_drift_reference(context_vector_batch=context_vec, y_batch=y_val, alpha=EMA_ALPHA)
                
            # Lưu checkpoint
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"adapted_{chunk_name}.pth"))
            
            # ========== BƯỚC 4: CẬP NHẬT REPLAY BUFFER ==========
            for x_buf, y_buf in DataLoader(chunk_ds, batch_size=256, shuffle=True):
                replay_buffer.add_samples(x_buf, y_buf) # Lưu toàn bộ chuỗi
                break  # Chỉ lấy 1 batch đại diện
            
            # ========== GHI METRICS ==========
            with open(METRICS_CSV, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([step_counter, phase_label, chunk_name, f"{f1_chunk:.4f}", 
                                f"{s_rep:.6f}", f"{s_kan:.6f}", f"{avg_entropy:.4f}", drift_type, action])
            
            # Giải phóng RAM chunk hiện tại
            del chunk_df, chunk_ds, chunk_loader
    
    # ===================================================================
    #                    KẾT THÚC - BÁO CÁO TỔNG KẾT
    # ===================================================================
    logging.info("\n" + "🎉" * 20)
    logging.info("🎉 ĐÃ HOÀN TẤT TOÀN BỘ LUỒNG CONTINUAL LEARNING!")
    logging.info("🎉" * 20)
    logging.info(f"📊 File metrics đã lưu tại: {METRICS_CSV}")
    logging.info(f"💾 Checkpoints đã lưu tại: {CHECKPOINT_DIR}")
    logging.info(f"📝 Log chi tiết tại: {LOG_FILE}")
    
    # Lưu mô hình cuối cùng
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    logging.info("💾 Đã lưu mô hình cuối cùng.")

if __name__ == "__main__":
    main()
