import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes:int, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim()-1)).float()
        pred = pred.contiguous().view(pred.size(0), self.num_classes, -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.size(0), self.num_classes, -1)

        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2.*intersection+self.smooth)/(union+self.smooth)
        loss = 1 - dice.mean()
        
        return loss
    
class MixLoss(nn.Module):
    def __init__(self, num_classes:int, entropy_weight:float, entropy_label_smoothing=0, dice_smooth=1e-5):
        super().__init__()
        self.entropy_loss = nn.CrossEntropyLoss(label_smoothing=entropy_label_smoothing)
        self.dice_loss = DiceLoss(num_classes, smooth=dice_smooth)
        self.entropy_weight = entropy_weight

    def forward(self, pred, target):
        loss1 = self.entropy_loss(pred, target)
        loss2 = self.dice_loss(pred, target)
        return self.entropy_weight*loss1 + (1-self.entropy_weight)*loss2