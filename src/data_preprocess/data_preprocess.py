import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ================= ĐƯỜNG DẪN ĐỘNG =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
SRC_DIR = os.path.dirname(CURRENT_DIR)                  
PROJECT_ROOT = os.path.dirname(SRC_DIR)                  

RAW_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, 'preprocess_geokan.log'), mode='w', encoding='utf-8'),
                              logging.StreamHandler()])

DATASET_PHASES = {
    "Phase1_Train": ["CICIoT2023", "CICDDoS2019Full"],
    "Phase2_DriftTest": ["CSE-CIC-IDS-2018-Full", "CICDarknet2020CSVs", "CICDDoS2017", "CIC-DoHBrw-2020"],
    "Phase3_DomainShift": ["CICEVSE2024_Dataset"]
}

COLUMN_MAP_DICT = {
    'flow_duration': 'Flow Duration', 'duration': 'Flow Duration', 'flow.duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets', 'total_fwd_packets': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Bwd Packets', 'total_bwd_packets': 'Total Bwd Packets',
    'flow_pkts_s': 'Flow Packets/s', 'rate': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean', 'fwd_header_len': 'Fwd Header Length', 'header_length': 'Fwd Header Length',
    'label': 'Label', 'traffic category': 'Label', 'doh': 'Label', 'class': 'Label', 'event_type': 'Label',
    'fin_flag_number': 'FIN Flag Count', 'syn_flag_number': 'SYN Flag Count',
    'rst_flag_number': 'RST Flag Count', 'psh_flag_number': 'PSH Flag Count',
    'ack_flag_number': 'ACK Flag Count', 'urg_flag_number': 'URG Flag Count', 
    'max': 'Max Packet Length', 'min': 'Min Packet Length', 
    'mean': 'Packet Length Mean', 'std': 'Packet Length Std'
}

# ================= TỪ ĐIỂN QUY HOẠCH NHÃN (LABEL MAPPING) =================
# LABEL_MAPPING = 
# {
#     # 1. Nhóm Bình thường (Benign)
#     'benign': 'Benign', 'normal': 'Benign', '0': 'Benign',
    
#     # 2. Nhóm DoS (Từ chối dịch vụ)
#     'dos hulk': 'DoS', 'dos goldeneye': 'DoS', 'dos slowloris': 'DoS', 'dos slowhttptest': 'DoS',
#     'dos attacks-hulk': 'DoS', 'dos attacks-goldeneye': 'DoS', 'dos attacks-slowloris': 'DoS',
    
#     # 3. Nhóm DDoS (Từ chối dịch vụ phân tán)
#     'ddos': 'DDoS', 'ddos attacks-loic-http': 'DDoS', 'ddos attacks-hoic': 'DDoS', 
#     'ddos_udp': 'DDoS', 'ddos_icmp': 'DDoS', 'syn': 'DDoS', 'udp': 'DDoS', 'udplag': 'DDoS', 
#     'ldap': 'DDoS', 'netbios': 'DDoS', 'mssql': 'DDoS', 'portmap': 'DDoS',
    
#     # 4. Nhóm Brute Force (Dò mật khẩu)
#     'ftp-patator': 'Brute_Force', 'ssh-patator': 'Brute_Force', 'bruteforce': 'Brute_Force',
#     'brute force -web': 'Brute_Force', 'brute force -xss': 'Brute_Force', 'ftp-bruteforce': 'Brute_Force',
    
#     # 5. Nhóm Web Attack (Tấn công Web)
#     'web attack \x96 brute force': 'Web_Attack', 'web attack \x96 xss': 'Web_Attack', 
#     'web attack \x96 sqli': 'Web_Attack', 'sql injection': 'Web_Attack', 'web attack': 'Web_Attack',
    
#     # 6. Nhóm Botnet
#     'bot': 'Botnet', 'botnet': 'Botnet',
    
#     # 7. Nhóm Infiltration (Xâm nhập)
#     'infiltration': 'Infiltration',
    
#     # 8. Nhóm Port Scan (Quét cổng)
#     'portscan': 'PortScan'
# }
# ================= TỪ ĐIỂN QUY HOẠCH NHÃN VẠN NĂNG (Bao trùm 2017, 2018, 2019, 2023) =================
LABEL_MAPPING = {
    # 1. Nhóm Bình thường
    'benign': 'Benign', 'normal': 'Benign', '0': 'Benign',
    
    # 2. Nhóm DDoS (LƯU Ý QUAN TRỌNG: Phải đặt DDoS trước DoS để tránh từ 'dos' ăn mất chữ 'ddos')
    'ddos': 'DDoS', 'syn': 'DDoS', 'udp': 'DDoS', 'udplag': 'DDoS', 'ldap': 'DDoS', 
    'netbios': 'DDoS', 'mssql': 'DDoS', 'portmap': 'DDoS',
    
    # 3. Nhóm DoS (Rút gọn từ khóa để tóm được cả 'dos-udp_flood' và 'dos hulk')
    'dos': 'DoS', 'hulk': 'DoS', 'goldeneye': 'DoS', 'slowloris': 'DoS', 'slowhttptest': 'DoS',
    
    # 4. Nhóm Botnet (Thêm các họ Botnet nổi tiếng của IoT 2023)
    'bot': 'Botnet', 'mirai': 'Botnet', 'gafgyt': 'Botnet',
    
    # 5. Nhóm Web Attack (Thêm các lỗ hổng cụ thể)
    'web': 'Web_Attack', 'sql': 'Web_Attack', 'xss': 'Web_Attack', 'commandinjection': 'Web_Attack', 'browserhijacking': 'Web_Attack',
    
    # 6. Nhóm Brute Force
    'brute': 'Brute_Force', 'patator': 'Brute_Force',
    
    # 7. Nhóm Port Scan & Reconnaissance (Mở rộng để quét cả do thám mạng)
    'scan': 'PortScan', 'recon': 'PortScan', 'sweep': 'PortScan', 'discovery': 'PortScan',
    
    # 8. Nhóm Infiltration & Tấn công nội mạng
    'infiltration': 'Infiltration', 'spoofing': 'Infiltration', 'mitm': 'Infiltration'
}

def clean_and_map_label(label):
    # 1. Chuyển về chữ thường, xóa khoảng trắng thừa
    lbl = str(label).strip().lower()
    
    # 2. CHỈ thay thế các ký tự lỗi font cụ thể (KHÔNG dùng replace('', '-'))
    lbl = lbl.replace('\x96', '-') 
    
    # 3. Duyệt tìm từ khóa trong Label Mapping
    for key, mapped_val in LABEL_MAPPING.items():
        if key in lbl: # Kiểm tra xem từ khóa có nằm trong nhãn thô không
            return mapped_val
            
    return 'Other_Attack' # Chỉ trả về cái này nếu thực sự không tìm thấy gì

def normalize_columns(df):
    new_cols = []
    for col in df.columns:
        c_clean = str(col).strip()
        c_lower = c_clean.lower()
        if c_clean in COLUMN_MAP_DICT: final_name = COLUMN_MAP_DICT[c_clean]
        elif c_lower in COLUMN_MAP_DICT: final_name = COLUMN_MAP_DICT[c_lower]
        else: final_name = c_clean  # <--- Đổi thành c_clean để vĩnh viễn xóa dấu cách thừa
        new_cols.append(final_name)
    df.columns = new_cols
    return df.loc[:, ~df.columns.duplicated()]

def process_and_save_dataset(phase_name, dataset_name):
    raw_path = os.path.join(RAW_DIR, dataset_name)
    save_path = os.path.join(PROCESSED_DIR, phase_name, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(raw_path): return None

    all_files = []
    for root, dirs, filenames in os.walk(raw_path):
        for f in filenames:
            if f.lower().endswith(('.csv', '.parquet')):
                all_files.append(os.path.join(root, f))
                
    if not all_files: return None
        
    logging.info(f"\n🚀 ĐANG XỬ LÝ: {phase_name} -> {dataset_name} ({len(all_files)} files)")
    total_rows = 0
    dataset_label_counts = Counter()
    
    for full_path in tqdm(all_files, desc=f"Processing {dataset_name}", unit="file", dynamic_ncols=True):
        filename = os.path.basename(full_path)
        
        if os.path.getsize(full_path) == 0: continue
            
        try:
            df = pd.read_csv(full_path, encoding='latin1', on_bad_lines='skip', low_memory=False) if full_path.lower().endswith('.csv') else pd.read_parquet(full_path)
            if df.empty: continue
                
            df = normalize_columns(df)
            
            # --- 1. TÌM VÀ ĐỔI TÊN CỘT NHÃN ---
            if 'Label' not in df.columns:
                for c in df.columns:
                    if c.lower() in ['label', 'class', 'traffic category', 'attack_type', 'event_type']:
                        df.rename(columns={c: 'Label'}, inplace=True)
                        break
            if 'Label' not in df.columns: continue

            # --- 2. SẮP XẾP THỜI GIAN TRƯỚC KHI XÓA (TIME-SERIES INTEGRITY) ---
            time_cols = [c for c in df.columns if c.lower() in ['timestamp', 'time']]
            if time_cols:
                try:
                    df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors='coerce')
                    df = df.sort_values(by=time_cols[0]).reset_index(drop=True)
                except:
                    pass

            # --- 3. XÓA CỘT THỪA ---
            cols_to_drop = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 
                            'SimillarHTTP', 'SimilarHTTP', 'Unnamed: 0', 'Inbound']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
            
            # --- 4. MAP NHÃN VÀ TÁCH X, y ---
            df['Label'] = df['Label'].apply(clean_and_map_label)
            y = df['Label'].values
            X = df.drop(columns=['Label'])
            
            # --- 5. XỬ LÝ ĐẶC TRƯNG THÔNG MINH ---
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        # Thử ép kiểu sang số thực
                        X[col] = pd.to_numeric(X[col], errors='raise')
                    except ValueError:
                        # Nếu là chữ (vd: 'TCP', 'UDP'), thì chuyển thành số phân loại thay vì xóa thành NaN
                        X[col] = X[col].astype('category').cat.codes
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.dropna(axis=1, how='all', inplace=True) 
            X.fillna(0, inplace=True) 
            
            if X.empty: continue
            
            # Khởi tạo lại DataFrame sạch
            df_processed = pd.DataFrame(X, columns=X.columns)
            df_processed['Label'] = y
            
            dataset_label_counts.update(y)
            total_rows += len(df_processed)
            
            # Lưu file dạng Parquet cho tốc độ I/O cực nhanh
            safe_filename = filename.lower().replace('.csv', '.parquet')
            out_file = os.path.join(save_path, safe_filename)
            df_processed.to_parquet(out_file, engine='pyarrow', index=False)
            
        except pd.errors.EmptyDataError: continue
        except Exception as e:
            logging.error(f"Lỗi tại file {filename}: {e}")
            continue
            
    logging.info(f"✅ Hoàn thành {dataset_name} - Tổng: {total_rows:,} mẫu hợp lệ.")
    return dataset_label_counts

if __name__ == "__main__":
    logging.info("🔥 BẮT ĐẦU CHIẾN DỊCH TIỀN XỬ LÝ TIME SERIES (CÓ BẢO TOÀN ĐẶC TRƯNG & MAP NHÃN) 🔥")
    global_counts = Counter()
    
    for phase_name, datasets in DATASET_PHASES.items():
        logging.info(f"\n{'='*50}\n🌟 GIAI ĐOẠN: {phase_name}\n{'='*50}")
        for ds in datasets:
            ds_counts = process_and_save_dataset(phase_name, ds)
            if ds_counts:
                global_counts.update(ds_counts)
                
    logging.info("\n🎉 TOÀN BỘ QUÁ TRÌNH TIỀN XỬ LÝ ĐÃ HOÀN TẤT! 🎉")
    
    stats_df = pd.DataFrame(global_counts.most_common(), columns=['Label_Name', 'Total_Samples'])
    stats_csv_path = os.path.join(PROCESSED_DIR, 'global_label_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    
    logging.info(f"📊 Báo cáo số lượng Nhãn đã lưu tại: {stats_csv_path}")
    logging.info("👉 HÃY MỞ FILE BÁO CÁO TRÊN ĐỂ KIỂM TRA XEM CÓ CÒN 51 NHÃN KHÔNG NHÉ!")