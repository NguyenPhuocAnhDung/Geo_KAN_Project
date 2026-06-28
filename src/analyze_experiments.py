import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
METRICS_CSV = os.path.join(PROJECT_DIR, 'reports', 'ContinualStream_DriftTKAN', 'stream_metrics.csv')

def analyze_lead_time(df, f1_drop_threshold=0.10, window_size=5):
    """
    Phân tích Lead Time: 
    Tìm thời điểm Drift Detected (S_rep hoặc S_KAN cảnh báo) và đo lường khoảng 
    cách (số chunk) so với thời điểm F1-score thực sự "gãy" (giảm > threshold 
    so với trung bình window trước đó).
    """
    print(f"\n========== PHÂN TÍCH LEAD TIME (Ngưỡng sụt giảm F1 = {f1_drop_threshold*100}%) ==========")
    
    # 1. Tìm các chunk có báo động Drift
    drift_chunks = df[df['Drift_Type'] != 'No Drift'].index.tolist()
    
    if not drift_chunks:
        print("Không có báo động Drift nào trong toàn bộ quá trình stream.")
        return
        
    # 2. Tìm thời điểm F1 sụp đổ (Collapse)
    f1_collapse_chunks = []
    for i in range(window_size, len(df)):
        past_f1_mean = df['F1_Macro'].iloc[i-window_size:i].mean()
        curr_f1 = df['F1_Macro'].iloc[i]
        
        if (past_f1_mean - curr_f1) >= f1_drop_threshold:
            f1_collapse_chunks.append((i, past_f1_mean, curr_f1))
            
    if not f1_collapse_chunks:
        print("Không có sự kiện sụp đổ F1 nào được ghi nhận.")
        return
        
    # 3. So khớp từng cú sụp đổ với Cảnh báo sớm nhất trước đó
    for collapse_idx, past_f1, curr_f1 in f1_collapse_chunks:
        chunk_name = df['Chunk'].iloc[collapse_idx]
        
        # Tìm cảnh báo gần nhất TRƯỚC cú sụp đổ
        prior_drifts = [idx for idx in drift_chunks if idx < collapse_idx]
        
        if prior_drifts:
            earliest_warning = prior_drifts[0]  # Lấy báo động đầu tiên của đợt này
            lead_time = collapse_idx - earliest_warning
            
            warning_chunk = df['Chunk'].iloc[earliest_warning]
            drift_type = df['Drift_Type'].iloc[earliest_warning]
            
            print(f"- Sụp đổ tại Chunk: {chunk_name} (F1 giảm từ {past_f1:.2f} xuống {curr_f1:.2f})")
            print(f"  + Cảnh báo sớm tại: {warning_chunk} ({drift_type})")
            print(f"  + Lead-time (Độ trễ): {lead_time} chunks")
            print("-" * 50)
            
            # Xóa các cảnh báo thuộc về đợt này để tránh lặp
            drift_chunks = [idx for idx in drift_chunks if idx > collapse_idx]
        else:
            print(f"- Sụp đổ tại Chunk: {chunk_name}. KHÔNG CÓ BÁO ĐỘNG (Missed Drift)!")
            print("-" * 50)

def analyze_correlation(df):
    """
    Phân tích Tương quan (Experiment E):
    Chứng minh S_rep, S_KAN, và Entropy thực sự có tương quan (âm) với F1-score.
    """
    print(f"\n========== PHÂN TÍCH TƯƠNG QUAN (Pearson & Spearman) ==========")
    metrics = ['S_rep', 'S_KAN', 'Entropy']
    target = 'F1_Macro'
    
    for metric in metrics:
        # Nếu cột chưa có hoặc toàn 0, bỏ qua
        if df[metric].std() == 0:
            continue
            
        pearson_corr, p_val_p = pearsonr(df[metric], df[target])
        spearman_corr, p_val_s = spearmanr(df[metric], df[target])
        
        print(f"{metric} vs {target}:")
        print(f"  - Pearson r  : {pearson_corr:.4f} (p-value: {p_val_p:.4e})")
        print(f"  - Spearman r : {spearman_corr:.4f} (p-value: {p_val_s:.4e})")
        
def main():
    if not os.path.exists(METRICS_CSV):
        print(f"❌ Không tìm thấy file metrics: {METRICS_CSV}")
        return
        
    df = pd.read_csv(METRICS_CSV)
    
    # Lọc bỏ Phase 1 (Train)
    df_stream = df[df['Phase'] != 'Phase1_Train'].copy().reset_index(drop=True)
    
    if len(df_stream) < 10:
        print("Dữ liệu quá ngắn để phân tích.")
        return
        
    # Chạy Experiment D: Lead Time Analysis
    analyze_lead_time(df_stream, f1_drop_threshold=0.10)
    analyze_lead_time(df_stream, f1_drop_threshold=0.15) # Sensitivity Analysis
    
    # Chạy Experiment E: Correlation Analysis
    analyze_correlation(df_stream)

if __name__ == "__main__":
    main()
