"""
Feature Pyramid Network (FPN) for multi-scale feature fusion
Critical for small object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import build_attention


class FPN(nn.Module):
    """
    Feature Pyramid Network
    Creates a multi-scale feature pyramid for detecting objects at different sizes
    """
    def __init__(
        self,
        in_channels_list,  # [C2, C3, C4, C5] channels
        out_channels=256,
        num_outs=5,
        attention_type='cbam'
    ):
        super(FPN, self).__init__()
        self.num_outs = num_outs

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )

        # Output convolutions (3x3 conv to merge features)
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Attention modules for enhanced feature representation
        self.attention_modules = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.attention_modules.append(
                build_attention(attention_type, out_channels)
            )

        # Extra layers for P6 and P7 (for very large images)
        if num_outs > len(in_channels_list):
            self.extra_convs = nn.ModuleList()
            for i in range(num_outs - len(in_channels_list)):
                self.extra_convs.append(
                    nn.Conv2d(
                        out_channels if i == 0 else out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1
                    )
                )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: dict with keys {C2, C3, C4, C5} from backbone

        Returns:
            list: Multi-scale features [P2, P3, P4, P5, (P6)]
        """
        # Convert dict to list in order
        feature_list = [features['C2'], features['C3'], features['C4'], features['C5']]

        # Build lateral connections
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(feature_list[i]))

        # Top-down pathway with feature fusion
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
            # Add to lower-level feature
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions and attention
        outs = []
        for i in range(len(laterals)):
            x = self.fpn_convs[i](laterals[i])
            x = self.attention_modules[i](x)
            outs.append(x)

        # Add extra levels if needed (P6, P7)
        if self.num_outs > len(outs):
            for i in range(self.num_outs - len(outs)):
                if i == 0:
                    outs.append(self.extra_convs[i](outs[-1]))
                else:
                    outs.append(self.extra_convs[i](F.relu(outs[-1])))

        return outs


class PAN(nn.Module):
    """
    Path Aggregation Network (PANet)
    Enhances FPN with bottom-up path augmentation
    Better for small object detection
    """
    def __init__(
        self,
        in_channels_list,
        out_channels=256,
        attention_type='cbam'
    ):
        super(PAN, self).__init__()

        # FPN part (top-down)
        self.fpn = FPN(
            in_channels_list,
            out_channels,
            num_outs=len(in_channels_list),
            attention_type=attention_type
        )

        # Bottom-up path augmentation
        self.downsample_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.downsample_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            )
            self.pan_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        """
        Args:
            features: dict with keys {C2, C3, C4, C5}

        Returns:
            list: Enhanced multi-scale features
        """
        # Top-down pathway (FPN)
        fpn_outs = self.fpn(features)

        # Bottom-up pathway
        pan_outs = [fpn_outs[0]]
        for i in range(len(fpn_outs) - 1):
            # Downsample previous level
            downsampled = self.downsample_convs[i](pan_outs[-1])
            # Fuse with FPN feature
            fused = downsampled + fpn_outs[i + 1]
            # Apply convolution
            pan_outs.append(self.pan_convs[i](fused))

        return pan_outs


if __name__ == "__main__":
    # Test FPN
    features = {
        'C2': torch.randn(2, 256, 128, 128),
        'C3': torch.randn(2, 512, 64, 64),
        'C4': torch.randn(2, 1024, 32, 32),
        'C5': torch.randn(2, 2048, 16, 16)
    }

    fpn = FPN([256, 512, 1024, 2048], out_channels=256, num_outs=5)
    outs = fpn(features)
    for i, out in enumerate(outs):
        print(f"P{i+2}: {out.shape}")

    # Test PAN
    pan = PAN([256, 512, 1024, 2048], out_channels=256)
    outs = pan(features)
    for i, out in enumerate(outs):
        print(f"PAN_P{i+2}: {out.shape}")
