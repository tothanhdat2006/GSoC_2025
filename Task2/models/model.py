from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet50_extractor(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(resnet.children())[:-2]).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feature = self.model(x)
        return feature.reshape(feature.shape[0], -1)
    
class Resnet101_extractor(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(resnet.children())[:-2]).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feature = self.model(x)
        return feature.reshape(feature.shape[0], -1)


def byol_loss(q, z):
    """
    BYOL loss from Eq. 2 in the paper
    """
    q = F.normalize(q, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (q * z).sum(dim=-1).mean()
