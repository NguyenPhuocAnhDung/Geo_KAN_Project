import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import pyarrow.parquet as pq
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed", "Phase1_Train")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "global_scaler.pkl")

def build_global_scaler():
    all_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
    if not all_files: return logging.error("❌ Không tìm thấy file Parquet nào!")

    # --- 1. QUÉT TÌM 'SIÊU ĐẶC TRƯNG' BAO TRÙM TẤT CẢ DATASET ---
    logging.info("🔍 Đang quét để hợp nhất các cột đặc trưng (Union Features)...")
    all_features = set()
    for f in tqdm(all_files, desc="Quét Cột", dynamic_ncols=True):
        cols = pq.ParquetFile(f).schema.names
        all_features.update([c for c in cols if c != 'Label'])
        
    global_features = sorted(list(all_features))
    logging.info(f"🎯 Đã hợp nhất thành công {len(global_features)} đặc trưng duy nhất.")

    # --- 2. FIT SCALER VỚI DANH SÁCH ĐÃ HỢP NHẤT ---
    scaler = StandardScaler()
    logging.info(f"🚀 BẮT ĐẦU TẠO KHUÔN SCALER MỚI...")

    for file_path in tqdm(all_files, desc="Fit Scaler", dynamic_ncols=True):
        try:
            df = pq.read_table(file_path).to_pandas()
            X = df.drop(columns=['Label'], errors='ignore')
            
            # Căn chỉnh: Ép DataFrame hiện tại khớp hoàn toàn với global_features (thiếu thì điền 0)
            X = X.reindex(columns=global_features, fill_value=0.0)
            scaler.partial_fit(X)
        except Exception as e:
            logging.error(f"Lỗi tại file {file_path}: {e}")

    # Gắn trực tiếp danh sách cột vào Scaler để file Train dùng lại
    scaler.feature_names_in_ = np.array(global_features)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"✅ Đã lưu Scaler mới tại: {SCALER_PATH}")

if __name__ == "__main__":
    build_global_scaler()