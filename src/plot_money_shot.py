import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. ĐỌC DỮ LIỆU TỪ LOGS ---
# Đường dẫn file
baseline_csv = 'reports/ContinualStream_DriftTKAN/logs_static_baseline.csv'
adaptive_csv = 'reports/ContinualStream_DriftTKAN/logs_proposed_adaptive.csv'

# Nếu chưa có đủ file, tạo giả lập để hiển thị format mẫu
if os.path.exists(baseline_csv) and os.path.exists(adaptive_csv):
    df_static = pd.read_csv(baseline_csv)
    df_adaptive = pd.read_csv(adaptive_csv)
    
    # Lấy data (chỉ vẽ từ Phase 2 và 3 để dễ nhìn, hoặc vẽ toàn bộ)
    time_chunks = np.arange(1, len(df_static) + 1)
    f1_static = df_static['f1_macro'].values
    f1_adaptive = df_adaptive['f1_macro'].values
    shift_score = df_adaptive['shift_score'].values
    phases = df_adaptive['phase'].values
else:
    print("⚠️ Cảnh báo: Chưa có đủ file CSV. Đang hiển thị bằng Dữ Liệu Giả Lập để Preview!")
    time_chunks = np.arange(1, 25) 
    f1_static = np.concatenate([np.random.normal(0.82, 0.02, 8), np.random.normal(0.02, 0.01, 8), np.random.normal(0.22, 0.05, 8)])
    f1_adaptive = np.concatenate([np.random.normal(0.82, 0.02, 8), np.random.normal(0.78, 0.03, 8), np.random.normal(0.75, 0.04, 8)])
    shift_score = np.concatenate([np.random.normal(0.001, 0.0005, 8), np.random.normal(0.03, 0.005, 8), np.random.normal(0.044, 0.004, 8)])
    phases = np.array(['Phase1'] * 8 + ['Phase2'] * 8 + ['Phase3'] * 8)

DRIFT_THRESHOLD = 0.025

# --- CẤU HÌNH ĐỒ THỊ CHUẨN IEEE ---
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

# --- Trục trái: F1 Score (Màu xanh/Xám) ---
ax1.set_xlabel('Data Stream (Chronological Time Chunks)', fontweight='bold')
ax1.set_ylabel('F1-Macro Score', color='black', fontweight='bold')
line1, = ax1.plot(time_chunks, f1_static, color='tab:gray', linestyle='--', linewidth=2, label='Static TKAN (No Adaptation Baseline)')
line2, = ax1.plot(time_chunks, f1_adaptive, color='tab:blue', marker='o', linewidth=2.5, label='Adaptive TKAN (Ours)')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0.0, 1.05)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Trục right: Shift Score (Màu đỏ) ---
ax2 = ax1.twinx()  
ax2.set_ylabel('KAN Shift Score (Concept Drift)', color='tab:red', fontweight='bold')
line3, = ax2.plot(time_chunks, shift_score, color='tab:red', marker='s', linestyle='-', alpha=0.8, label='White-box Shift Score')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Vẽ đường Threshold
threshold_line = ax2.axhline(y=DRIFT_THRESHOLD, color='black', linestyle=':', linewidth=2, label=f'Detection Threshold ($\\tau \\approx {DRIFT_THRESHOLD}$)')

# --- ĐÁNH DẤU CÁC ĐIỂM FINE-TUNE (ADAPTATION) ---
drift_detected = shift_score > DRIFT_THRESHOLD
ax2.scatter(time_chunks[drift_detected], shift_score[drift_detected], color='gold', s=150, edgecolor='black', zorder=5, marker='*')

# --- PHÂN VÙNG CÁC PHASE DỰA TRÊN DỮ LIỆU THỰC TẾ ---
# Tìm các điểm chuyển giao Phase để tô màu
phase_changes = []
current_phase = phases[0]
for i, p in enumerate(phases):
    if p != current_phase:
        phase_changes.append(i)
        current_phase = p

p1_end = phase_changes[0] if len(phase_changes) > 0 else len(time_chunks)//3
p2_end = phase_changes[1] if len(phase_changes) > 1 else 2*len(time_chunks)//3

ax1.axvspan(0, p1_end, facecolor='tab:green', alpha=0.08)
ax1.text(p1_end / 2, 0.1, 'Phase 1: Stable\n(CIC-IDS 2017/18)', ha='center', va='center', fontweight='bold', color='darkgreen')

ax1.axvspan(p1_end, p2_end, facecolor='tab:orange', alpha=0.08)
ax1.text(p1_end + (p2_end - p1_end) / 2, 0.1, 'Phase 2: Drift 1\n(CICIoT 2023)', ha='center', va='center', fontweight='bold', color='darkorange')

ax1.axvspan(p2_end, len(time_chunks), facecolor='tab:red', alpha=0.08)
ax1.text(p2_end + (len(time_chunks) - p2_end) / 2, 0.1, 'Phase 3: Drift 2\n(CICEVSE 2024)', ha='center', va='center', fontweight='bold', color='darkred')

# --- GOM LEGENDS TỰ ĐỘNG ---
lines = [line1, line2, line3, threshold_line]
labels = [l.get_label() for l in lines]
import matplotlib.lines as mlines
star_marker = mlines.Line2D([], [], color='white', marker='*', markerfacecolor='gold', markeredgecolor='black', markersize=12, label='Adaptation Triggered')
lines.append(star_marker)
labels.append('Adaptation Triggered')

ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.title('Performance and White-box Concept Drift Detection over Continuous Data Stream', fontweight='bold', pad=15)
plt.tight_layout()

# Tạo thư mục và lưu ảnh
os.makedirs('reports/plots', exist_ok=True)
plt.savefig('reports/plots/money_shot_drift.pdf', format='pdf', bbox_inches='tight')
plt.savefig('reports/plots/money_shot_drift.png', format='png', dpi=300, bbox_inches='tight')
print("✅ Đã xuất biểu đồ ra file: reports/plots/money_shot_drift.pdf (và .png)")

# plt.show()
