"""
Attention modules for enhanced small object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Enhances feature representation for small objects
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention
    Lightweight alternative to CBAM
    """
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def build_attention(attention_type, in_channels, **kwargs):
    """Factory function for attention modules"""
    if attention_type == 'cbam':
        return CBAM(in_channels, **kwargs)
    elif attention_type == 'se':
        return SEAttention(in_channels, **kwargs)
    elif attention_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Test CBAM
    x = torch.randn(2, 256, 64, 64)
    cbam = CBAM(256)
    out = cbam(x)
    print(f"CBAM output: {out.shape}")

    # Test SE
    se = SEAttention(256)
    out = se(x)
    print(f"SE output: {out.shape}")
