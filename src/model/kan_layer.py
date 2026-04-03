import torch
import torch.nn as nn

class ChebyshevKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super(ChebyshevKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        self.cheb_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        nn.init.normal_(self.cheb_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))

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