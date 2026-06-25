import torch
import torch.nn as nn
import torch.nn.functional as F
from model.kan_layer import ChebyshevKANLayer 

# ================= MÔ HÌNH CŨ (GIỮ LẠI LÀM KỶ NIỆM) =================
class Hybrid_TKAN(nn.Module):
    def __init__(self, input_features, num_classes, lstm_hidden=64, mlp_hidden=32, cheb_degree=3):
        super(Hybrid_TKAN, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # <--- BiLSTM
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden), 
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU()
        )
        
        self.kan_classifier = ChebyshevKANLayer(
            in_features=mlp_hidden, 
            out_features=num_classes, 
            degree=cheb_degree
        )

    def forward(self, x):
        self.lstm.flatten_parameters() 
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] 
        
        mlp_features = self.mlp(last_step_out)
        logits = self.kan_classifier(mlp_features)
        
        return logits

# ================= MÔ HÌNH MỚI (BẢN TỐI THƯỢNG) =================
class Attention_TKAN(nn.Module):
    def __init__(self, input_features, num_classes, lstm_hidden=64, mlp_hidden=32, cheb_degree=3):
        super(Attention_TKAN, self).__init__()
        
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
        
        return logits

# ================= KỊCH BẢN TEST VRAM/CUDA =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Đang sử dụng thiết bị tính toán: {device.type.upper()} 🔥")
    
    BATCH_SIZE = 128
    SEQ_LEN = 10
    FEATURES = 114  # Đổi thành 114 cho khớp thực tế
    CLASSES = 9     # Đổi thành 9 nhãn
    
    # Đã đổi sang test mô hình mới: Attention_TKAN
    model = Attention_TKAN(input_features=FEATURES, num_classes=CLASSES)
    model = model.to(device)
    print("✅ Đã khởi tạo và đẩy mô hình Attention_TKAN lên VRAM.")
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES).to(device)
    print(f"   Shape đầu vào (3D): {dummy_input.shape}")
    
    with torch.no_grad(): 
        output = model(dummy_input)
        
    print(f"✅ Chạy thành công! Shape đầu ra: {output.shape}")
    
    if torch.cuda.is_available():
        print(f"💾 VRAM đang tiêu thụ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")