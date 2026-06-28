import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
METRICS_CSV = os.path.join(PROJECT_DIR, 'reports', 'ContinualStream_DriftTKAN', 'stream_metrics.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'reports', 'ContinualStream_DriftTKAN', 'plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_timeline():
    if not os.path.exists(METRICS_CSV):
        print(f"❌ Không tìm thấy file {METRICS_CSV}")
        return
        
    df = pd.read_csv(METRICS_CSV)
    
    df_stream = df[df['Phase'] != 'Phase1_Train'].copy()
    
    if len(df_stream) == 0:
        print("⚠️ Chưa có dữ liệu stream để vẽ.")
        return
        
    df_stream = df_stream.sort_values(by='Step').reset_index(drop=True)
    df_stream['Time_Step'] = df_stream.index + 1
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(df_stream['Time_Step'], df_stream['F1_Macro'], color='green', marker='o', markersize=4, linestyle='-', linewidth=1.5, label='F1-Macro')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Hierarchical White-box Drift Sensing Timeline', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left')
    
    ax2 = axes[1]
    ax2.plot(df_stream['Time_Step'], df_stream['S_rep'], color='blue', marker='s', markersize=4, linestyle='-', linewidth=1.5, label='Feature Shift (S_rep)')
    ax2.set_ylabel('S_rep (Cosine Dist)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    
    ax3 = axes[2]
    ax3.plot(df_stream['Time_Step'], df_stream['S_KAN'], color='purple', marker='^', markersize=4, linestyle='-', linewidth=1.5, label='Decision Shift (S_KAN)')
    ax3.set_ylabel('S_KAN (Shift Score)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    
    ax4 = axes[3]
    ax4.plot(df_stream['Time_Step'], df_stream['Entropy'], color='orange', marker='d', markersize=4, linestyle='-', linewidth=1.5, label='Predictive Uncertainty (H)')
    ax4.set_ylabel('Confidence Entropy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time Step (Data Chunk)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    
    for i, row in df_stream.iterrows():
        if row['Drift_Type'] != 'No Drift':
            for ax in axes:
                ax.axvline(x=row['Time_Step'], color='red', linestyle=':', alpha=0.3)
                
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'drift_timeline.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ tại: {output_path}")

if __name__ == "__main__":
    plot_timeline()
