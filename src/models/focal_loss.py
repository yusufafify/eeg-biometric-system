import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation as a drop-in replacement for CrossEntropyLoss.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (torch.Tensor): Weighting factor for each class.
        gamma (float): Focusing parameter (down-weights easy examples).
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels (N,)
        """
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # p_t: Probability of the correct class
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Combined loss
        loss = focal_term * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            # Move alpha to same device as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Select weights for the specific target classes
            at = self.alpha.gather(0, targets.data)
            loss = at * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
