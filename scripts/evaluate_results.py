import pandas as pd
import numpy as np

# Đọc file kết quả
df = pd.read_csv('/data/quyhv/Geo_KAN_Project/reports/ContinualStream_DriftTKAN/stream_metrics.csv')

# Tính toán các chỉ số
print("=== THỐNG KÊ KẾT QUẢ ĐÁNH GIÁ (STREAM METRICS) ===")
print(f"Tổng số chunks (steps): {len(df)}")

# F1-macro qua các phase
print("\n1. F1-Macro trung bình theo Phase:")
phase_groups = df.groupby('phase')
for name, group in phase_groups:
    print(f" - {name}: Mean = {group['f1_macro'].mean():.4f}, Max = {group['f1_macro'].max():.4f}, Min = {group['f1_macro'].min():.4f}")

# Shift Score
print("\n2. Shift Score (Drift Detection):")
print(f" - Max Shift Score: {df['shift_score'].max():.6f}")
print(f" - Mean Shift Score: {df['shift_score'].mean():.6f}")
print(f" - Có bao nhiêu lần vượt ngưỡng 0.05? {df['drift_detected'].sum()} lần")

# Sự tiến hóa của Shift Score
phase2_df = df[df['phase'] == 'Phase2_DriftTest']
phase3_df = df[df['phase'] == 'Phase3_DomainShift']
if not phase2_df.empty:
    print(f"\n3. Tiến hóa Shift Score trong Phase 2:")
    print(f" - Đầu Phase 2: {phase2_df['shift_score'].iloc[0]:.6f}")
    print(f" - Cuối Phase 2: {phase2_df['shift_score'].iloc[-1]:.6f}")

if not phase3_df.empty:
    print(f"\n4. Shift Score khi chuyển sang Domain mới (Phase 3 - Xe điện):")
    print(f" - Đầu Phase 3: {phase3_df['shift_score'].iloc[0]:.6f}")
    print(f" - Cuối Phase 3: {phase3_df['shift_score'].iloc[-1]:.6f}")

