"""
Main Small Object Detector Model
Combines backbone, neck, and head
"""
import torch
import torch.nn as nn
from .backbones.resnet import ResNetBackbone
from .necks.fpn import FPN, PAN
from .heads.detection_head import DetectionHead, DecoupledHead


class SmallObjectDetector(nn.Module):
    """
    End-to-end small object detector
    Architecture: Backbone -> Neck (FPN/PAN) -> Detection Head
    """
    def __init__(self, config):
        super(SmallObjectDetector, self).__init__()
        self.config = config

        # Build backbone
        self.backbone = self._build_backbone()

        # Build neck (FPN/PAN)
        self.neck = self._build_neck()

        # Build detection head
        self.head = self._build_head()

    def _build_backbone(self):
        """Build backbone network"""
        backbone_name = self.config['model']['backbone']

        if 'resnet' in backbone_name:
            return ResNetBackbone(
                pretrained=self.config['model']['pretrained']
            )
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")

    def _build_neck(self):
        """Build neck network (FPN/PAN)"""
        if not self.config['model']['fpn']['enabled']:
            return nn.Identity()

        feature_channels = [256, 512, 1024, 2048]  # ResNet feature channels
        fpn_channels = self.config['model']['fpn']['channels']

        fusion_type = self.config['model']['feature_fusion']['type']
        attention_type = self.config['model']['attention']['type'] if \
                        self.config['model']['attention']['enabled'] else 'none'

        if fusion_type == 'panet':
            return PAN(
                in_channels_list=feature_channels,
                out_channels=fpn_channels,
                attention_type=attention_type
            )
        elif fusion_type == 'fpn':
            return FPN(
                in_channels_list=feature_channels,
                out_channels=fpn_channels,
                num_outs=5,
                attention_type=attention_type
            )
        else:
            raise NotImplementedError(f"Neck type {fusion_type} not implemented")

    def _build_head(self):
        """Build detection head"""
        fpn_channels = self.config['model']['fpn']['channels']
        num_classes = self.config['data']['num_classes']

        # Use decoupled head for better performance
        return DecoupledHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=1  # Anchor-free
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)

        Returns:
            dict: Detection outputs {cls_scores, bbox_preds, obj_scores}
        """
        # Backbone
        features = self.backbone(x)

        # Neck
        if isinstance(self.neck, nn.Identity):
            neck_features = list(features.values())
        else:
            neck_features = self.neck(features)

        # Detection head
        outputs = self.head(neck_features)

        return outputs

    def get_losses(self, outputs, targets):
        """
        Compute losses for training
        To be implemented with specific loss functions
        """
        # This will be implemented in the loss module
        pass

    def get_predictions(self, outputs, conf_threshold=0.25, iou_threshold=0.45):
        """
        Post-process outputs to get final predictions
        To be implemented with NMS and filtering
        """
        # This will be implemented in the inference module
        pass


def build_detector(config):
    """Factory function to build detector"""
    return SmallObjectDetector(config)


if __name__ == "__main__":
    # Test detector
    import yaml

    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['data']['num_classes'] = 80  # COCO dataset

    # Build model
    model = build_detector(config)

    # Test forward pass
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)

    print("Model outputs:")
    print(f"Number of feature levels: {len(outputs['cls_scores'])}")
    for i in range(len(outputs['cls_scores'])):
        print(f"Level {i}:")
        print(f"  cls_scores: {outputs['cls_scores'][i].shape}")
        print(f"  bbox_preds: {outputs['bbox_preds'][i].shape}")
        print(f"  obj_scores: {outputs['obj_scores'][i].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
