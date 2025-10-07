"""
Detection head for small object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Detection head with classification and box regression
    Optimized for small objects with multiple anchor scales
    """
    def __init__(
        self,
        in_channels=256,
        num_classes=80,
        num_anchors=3,
        prior_prob=0.01
    ):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared convolution layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cls_score = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            3,
            padding=1
        )

        # Box regression head
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,  # x, y, w, h
            3,
            padding=1
        )

        # Objectness head (confidence)
        self.obj_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.obj_score = nn.Conv2d(
            in_channels // 2,
            num_anchors,
            3,
            padding=1
        )

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        """Initialize weights with bias for focal loss"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Args:
            features: list of multi-scale features from FPN/PAN

        Returns:
            dict: {
                'cls_scores': list of classification scores,
                'bbox_preds': list of bbox predictions,
                'obj_scores': list of objectness scores
            }
        """
        cls_scores = []
        bbox_preds = []
        obj_scores = []

        for feature in features:
            # Shared feature extraction
            x = self.shared_conv(feature)

            # Classification
            cls_feat = self.cls_conv(x)
            cls_score = self.cls_score(cls_feat)
            cls_scores.append(cls_score)

            # Box regression
            bbox_feat = self.bbox_conv(x)
            bbox_pred = self.bbox_pred(bbox_feat)
            bbox_preds.append(bbox_pred)

            # Objectness
            obj_feat = self.obj_conv(x)
            obj_score = self.obj_score(obj_feat)
            obj_scores.append(obj_score)

        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'obj_scores': obj_scores
        }


class DecoupledHead(nn.Module):
    """
    Decoupled detection head (like YOLOX)
    Separates classification and regression branches completely
    Better performance for small objects
    """
    def __init__(
        self,
        in_channels=256,
        num_classes=80,
        num_anchors=1,  # Anchor-free
        width_mult=1.0
    ):
        super(DecoupledHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        hidden_channels = int(256 * width_mult)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )

        # Classification branch
        self.cls_convs = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, 1)

        # Regression branch
        self.reg_convs = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        self.reg_pred = nn.Conv2d(hidden_channels, 4, 1)

        # Objectness branch
        self.obj_pred = nn.Conv2d(hidden_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias for classification
        nn.init.constant_(self.cls_pred.bias, -4.6)  # focal loss initialization

    def forward(self, features):
        """
        Args:
            features: list of multi-scale features

        Returns:
            dict: detection outputs
        """
        cls_scores = []
        bbox_preds = []
        obj_scores = []

        for feature in features:
            # Stem
            x = self.stem(feature)

            # Classification branch
            cls_feat = self.cls_convs(x)
            cls_score = self.cls_pred(cls_feat)
            cls_scores.append(cls_score)

            # Regression branch
            reg_feat = self.reg_convs(x)
            bbox_pred = self.reg_pred(reg_feat)
            bbox_preds.append(bbox_pred)

            # Objectness
            obj_score = self.obj_pred(reg_feat)
            obj_scores.append(obj_score)

        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'obj_scores': obj_scores
        }


if __name__ == "__main__":
    # Test detection head
    features = [
        torch.randn(2, 256, 128, 128),
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16)
    ]

    head = DetectionHead(in_channels=256, num_classes=80)
    outputs = head(features)

    print("Standard Detection Head:")
    for i, (cls, bbox, obj) in enumerate(zip(
        outputs['cls_scores'],
        outputs['bbox_preds'],
        outputs['obj_scores']
    )):
        print(f"Level {i}: cls={cls.shape}, bbox={bbox.shape}, obj={obj.shape}")

    # Test decoupled head
    decoupled_head = DecoupledHead(in_channels=256, num_classes=80)
    outputs = decoupled_head(features)

    print("\nDecoupled Detection Head:")
    for i, (cls, bbox, obj) in enumerate(zip(
        outputs['cls_scores'],
        outputs['bbox_preds'],
        outputs['obj_scores']
    )):
        print(f"Level {i}: cls={cls.shape}, bbox={bbox.shape}, obj={obj.shape}")
