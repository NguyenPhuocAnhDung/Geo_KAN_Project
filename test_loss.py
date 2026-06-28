import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Stable focal loss
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        ce_loss = F.nll_loss(log_pt, targets, weight=self.alpha, reduction='none')
        focal_loss = ((1 - pt.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma * ce_loss).mean()
        return focal_loss

# Test
x = torch.randn(2, 5) * 100 # Large logits
y = torch.tensor([0, 4])
loss = FocalLoss()(x, y)
print("Loss:", loss.item())
