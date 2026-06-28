import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.kan_layer import ChebyshevKANLayer 

class HierarchicalDriftTKAN(nn.Module):
    def __init__(self, input_features, num_classes, lstm_hidden=64, mlp_hidden=32, cheb_degree=3):
        super(HierarchicalDriftTKAN, self).__init__()
        
        # 1. Trích xuất đặc trưng chuỗi thời gian hai chiều
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Cơ chế Self-Attention học nhịp độ gói tin (Phân biệt DoS/DDoS)
        self.attention = nn.Linear(lstm_hidden * 2, 1)
        
        # Prototype Memory lưu trọng tâm của từng class (Shape: [C, D])
        self.register_buffer('class_centroids', torch.zeros(num_classes, lstm_hidden * 2))
        self.class_centroids_initialized = False
        
        # 3. Lớp nén đặc trưng
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU()
        )
        
        # 4. Lớp phân loại KAN vẽ ranh giới phi tuyến tính
        self.kan_classifier = ChebyshevKANLayer(
            in_features=mlp_hidden, 
            out_features=num_classes, 
            degree=cheb_degree
        )

    def forward(self, x):
        self.lstm.flatten_parameters() 
        lstm_out, _ = self.lstm(x) # Shape: [Batch, Seq=10, 128]
        
        # Tính trọng số Attention và lấy Vector Ngữ cảnh
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # Shape: [Batch, 10, 1]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # Shape: [Batch, 128]
        
        mlp_features = self.mlp(context_vector)
        logits = self.kan_classifier(mlp_features)
        
        # Trả về cả logits và context_vector để phục vụ đo lường Representation Drift
        return logits, context_vector

    def set_adaptation_mode(self, is_drifting=True):
        """
        Bật/tắt chế độ thích nghi (Fine-tuning) khi có Concept Drift.
        - Nếu is_drifting = True: Đóng băng BiLSTM/MLP, chỉ train KAN & Attention để tiết kiệm VRAM.
        - Nếu is_drifting = False: Mở khóa toàn bộ mô hình (dùng ở Pha 1 - Khởi tạo).
        """
        if is_drifting:
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.mlp.parameters():
                param.requires_grad = False
            for param in self.attention.parameters():
                param.requires_grad = True
            for param in self.kan_classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True

    def compute_hierarchical_drift(self, context_vector_batch, logits_batch):
        """
        Hierarchical Drift Sensing:
        1. S_rep (Feature Shift): Soft Prototype Matching bằng Posterior Distribution
        2. S_KAN (Decision Shift): KAN Coefficient Drift
        """
        # --- LEVEL 2: KAN Drift (Decision Shift) ---
        s_kan = self.kan_classifier.compute_shift_score()
        
        # --- LEVEL 1: Representation Drift (Feature Shift) ---
        if not self.class_centroids_initialized:
            s_rep = 0.0
        else:
            # P_soft = sum(p_i * P_i)
            # context_vector_batch: [Batch, D]
            # logits_batch: [Batch, C]
            probs = F.softmax(logits_batch, dim=1).detach() # [Batch, C]
            class_centroids_ema = self.class_centroids.detach() # [C, D]
            
            # Nhân ma trận để ra Soft Prototype cho từng sample: [Batch, C] x [C, D] -> [Batch, D]
            soft_prototypes = torch.matmul(probs, class_centroids_ema)
            
            # Cosine similarity từng sample
            sim = F.cosine_similarity(context_vector_batch.detach(), soft_prototypes)
            s_rep = (1.0 - sim).mean().item()
            
        return s_rep, s_kan

    def update_drift_reference(self, context_vector_batch=None, y_batch=None, alpha=0.99):
        """Cập nhật các trọng số tham chiếu (EMA) cho KAN và Class Prototypes (Chỉ dùng Ground Truth)."""
        # 1. Cập nhật KAN Coeffs EMA
        self.kan_classifier.update_base_coeffs(alpha=alpha)
        
        # 2. Cập nhật Class-wise Prototypes (Ground-truth Update)
        if context_vector_batch is not None and y_batch is not None:
            with torch.no_grad():
                curr_vecs = context_vector_batch.detach()
                y_true = y_batch.detach()
                
                if not self.class_centroids_initialized:
                    # Khởi tạo nóng (Tạm thời lấy mean toàn batch gán cho tất cả, hoặc chỉ gán cho nhãn xuất hiện)
                    # Cách chuẩn: Khởi tạo tất cả bằng 0, sau đó update
                    self.class_centroids_initialized = True
                
                # Cập nhật riêng cho từng class xuất hiện trong batch
                unique_classes = torch.unique(y_true)
                for c in unique_classes:
                    mask = (y_true == c)
                    class_mean = curr_vecs[mask].mean(dim=0)
                    
                    if torch.sum(torch.abs(self.class_centroids[c])) == 0:
                        self.class_centroids[c].copy_(class_mean)
                    else:
                        self.class_centroids[c].mul_(alpha).add_(class_mean, alpha=1 - alpha)

# ================= KỊCH BẢN TEST VRAM/CUDA =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Đang sử dụng thiết bị tính toán: {device.type.upper()} 🔥")
    
    BATCH_SIZE = 128
    SEQ_LEN = 10
    FEATURES = 114  # Đổi thành 114 cho khớp thực tế
    CLASSES = 9     # Đổi thành 9 nhãn
    
    # Test mô hình HierarchicalDriftTKAN
    model = HierarchicalDriftTKAN(input_features=FEATURES, num_classes=CLASSES)
    model = model.to(device)
    print("✅ Đã khởi tạo và đẩy mô hình HierarchicalDriftTKAN lên VRAM.")
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES).to(device)
    print(f"   Shape đầu vào (3D): {dummy_input.shape}")
    
    with torch.no_grad(): 
        output, context = model(dummy_input)
        
    print(f"✅ Chạy thành công! Shape logits: {output.shape}, Shape context: {context.shape}")
    
    if torch.cuda.is_available():
        print(f"💾 VRAM đang tiêu thụ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")