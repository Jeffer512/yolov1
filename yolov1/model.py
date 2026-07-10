import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import B, C


class YOLOv1(nn.Module):
    """YOLOv1 model with a ResNet-50 backbone and a custom convolutional head.
    
    Supports selective parameter freezing of the backbone layers to facilitate
    two-stage transfer learning and fine-tuning.
    """
    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super().__init__()
        self.B = B
        self.C = C
        self.output_depth = B * 5 + C

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Exclude average pooling and fully connected layers from the backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Custom convolutional projection head
        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),     
            nn.BatchNorm2d(1024),   
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, self.output_depth, kernel_size=1),
        ) 
        
        # Freeze backbone parameters on initialization
        self.freeze_backbone(True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def freeze_backbone(self, freeze=True):
        """Configures the parameter gradients of the backbone.

        Args:
            freeze (bool): If True, disables gradients for the entire 
                backbone. If False, enables gradients only for the final 
                residual block (layer4).
        """
        if freeze:
            self.backbone.requires_grad_(False)
        else:
            self.backbone.requires_grad_(False)
            self.backbone[-1].requires_grad_(True)