"""
ResNet backbone for small object detection
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    ResNet50 backbone with multi-scale feature extraction
    Optimized for small object detection
    """
    def __init__(self, pretrained=True, frozen_stages=-1):
        super(ResNetBackbone, self).__init__()

        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        # Extract different stages
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.stage1 = resnet.layer1  # C2: 1/4 resolution
        self.stage2 = resnet.layer2  # C3: 1/8 resolution
        self.stage3 = resnet.layer3  # C4: 1/16 resolution
        self.stage4 = resnet.layer4  # C5: 1/32 resolution

        # Feature channels for each stage
        self.feature_channels = {
            'C2': 256,
            'C3': 512,
            'C4': 1024,
            'C5': 2048
        }

        # Freeze stages if needed
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages):
        """Freeze model stages"""
        if frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        for i in range(frozen_stages):
            stage = stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            dict: Multi-scale features {C2, C3, C4, C5}
        """
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return {
            'C2': c2,
            'C3': c3,
            'C4': c4,
            'C5': c5
        }


if __name__ == "__main__":
    # Test backbone
    model = ResNetBackbone(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    features = model(x)

    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
