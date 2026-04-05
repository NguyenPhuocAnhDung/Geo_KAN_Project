import torch
import torch.nn as nn
from model.kan_layer import ChebyshevKANLayer 

class Hybrid_TKAN(nn.Module):
    def __init__(self, input_features, num_classes, lstm_hidden=64, mlp_hidden=32, cheb_degree=3):
        super(Hybrid_TKAN, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # <--- Bật chế độ BiLSTM ở đây
        )
        
        self.mlp = nn.Sequential(
            # Thay lstm_hidden thành lstm_hidden * 2 ở tham số đầu tiên
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
        # Dọn dẹp bộ nhớ đệm chống Warning của PyTorch RNN
        self.lstm.flatten_parameters() 
        
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] 
        
        mlp_features = self.mlp(last_step_out)
        logits = self.kan_classifier(mlp_features)
        
        return logits

# ================= KỊCH BẢN TEST VRAM/CUDA =================
if __name__ == "__main__":
    # Tự động nhận diện GPU NVIDIA T4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Đang sử dụng thiết bị tính toán: {device.type.upper()} 🔥")
    
    # Thông số giả lập
    BATCH_SIZE = 128
    SEQ_LEN = 10
    FEATURES = 78  # Số cột dữ liệu thực tế của bạn
    CLASSES = 8    # Số nhãn phân loại
    
    # 1. Khởi tạo mô hình và bắn lên VRAM
    model = Hybrid_TKAN(input_features=FEATURES, num_classes=CLASSES)
    model = model.to(device)
    print("✅ Đã khởi tạo và đẩy mô hình Hybrid_TKAN lên VRAM.")
    
    # 2. Tạo dữ liệu giả lập và cũng bắn lên VRAM
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES).to(device)
    print(f"   Shape đầu vào (3D): {dummy_input.shape}")
    
    # 3. Chạy thử nghiệm (Forward Pass)
    with torch.no_grad(): # Tắt tính gradient để test cho nhẹ VRAM
        output = model(dummy_input)
        
    print(f"✅ Chạy thành công! Shape đầu ra: {output.shape}")
    
    if torch.cuda.is_available():
        print(f"💾 VRAM đang tiêu thụ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")