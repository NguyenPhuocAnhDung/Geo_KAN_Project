import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebyshevKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super(ChebyshevKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        self.cheb_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        nn.init.normal_(self.cheb_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))
        
        # Buffer để lưu trọng số tham chiếu (cho việc tính toán Shift Score)
        self.register_buffer('base_coeffs', torch.zeros_like(self.cheb_coeffs))
        self.base_coeffs_initialized = False

    def forward(self, x):
        x = torch.tanh(x) 
        
        cheb_polys = []
        cheb_polys.append(torch.ones_like(x, device=x.device)) 
        
        if self.degree > 0:
            cheb_polys.append(x)
            
        for n in range(1, self.degree):
            t_next = 2 * x * cheb_polys[n] - cheb_polys[n-1]
            cheb_polys.append(t_next)
            
        cheb_tensor = torch.stack(cheb_polys, dim=-1)
        y = torch.einsum('bid,oid->bo', cheb_tensor, self.cheb_coeffs)
        
        return y

    def update_base_coeffs(self, alpha=0.99):
        """Cập nhật trạng thái tham chiếu của cheb_coeffs theo Exponential Moving Average (EMA).
        alpha: Hệ số EMA (0 < alpha < 1). Giá trị lớn (0.99) giúp reference thay đổi từ từ, chống nhiễu.
        """
        with torch.no_grad():
            if not self.base_coeffs_initialized:
                self.base_coeffs.copy_(self.cheb_coeffs.detach())
                self.base_coeffs_initialized = True
            else:
                # EMA update
                self.base_coeffs.mul_(alpha).add_(self.cheb_coeffs.detach(), alpha=1 - alpha)

    def compute_shift_score(self):
        """Tính toán khoảng cách Cosine Distance (1 - Cosine Similarity) giữa 
        trọng số batch hiện tại và trọng số tham chiếu EMA.
        Dùng .detach().cpu() để tránh rò rỉ VRAM trong Continual Learning.
        """
        if not self.base_coeffs_initialized:
            return 0.0
            
        # Tách khỏi đồ thị tính toán và đẩy về CPU (tuỳ chọn) hoặc giữ nguyên device không grad
        curr_coeffs = self.cheb_coeffs.detach().view(-1)
        base_coeffs = self.base_coeffs.detach().view(-1)
        
        # Cosine distance = 1 - Cosine Similarity
        similarity = F.cosine_similarity(curr_coeffs.unsqueeze(0), base_coeffs.unsqueeze(0))
        shift = 1.0 - similarity.item()
            
        return shift