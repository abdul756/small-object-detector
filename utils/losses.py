"""
Loss functions for small object detection
Includes Focal Loss, GIoU Loss, and combined detection loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Especially useful for small object detection
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss
    Better for small objects than standard IoU
    """
    def __init__(self, reduction='mean'):
        super(GIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) predicted boxes [x1, y1, x2, y2]
            target_boxes: (N, 4) target boxes [x1, y1, x2, y2]
        """
        # Calculate intersection
        x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * \
                     torch.clamp(y2_inter - y1_inter, min=0)

        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)

        # Calculate enclosing box
        x1_enclose = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        y1_enclose = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        x2_enclose = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        y2_enclose = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (x2_enclose - x1_enclose) * (y2_enclose - y1_enclose)

        # Calculate GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

        # GIoU loss
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    Combined detection loss for small objects
    Includes:
    - Focal loss for classification
    - GIoU loss for bbox regression
    - BCE loss for objectness
    """
    def __init__(
        self,
        num_classes,
        bbox_loss_weight=1.0,
        cls_loss_weight=0.5,
        obj_loss_weight=1.0,
        small_object_weight=2.0,
        small_object_threshold=32
    ):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.small_object_weight = small_object_weight
        self.small_object_threshold = small_object_threshold

        # Loss functions
        self.focal_loss = FocalLoss()
        self.giou_loss = GIoULoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def compute_small_object_mask(self, target_boxes):
        """
        Compute mask for small objects
        Small objects get higher weight in loss
        """
        widths = target_boxes[:, 2] - target_boxes[:, 0]
        heights = target_boxes[:, 3] - target_boxes[:, 1]
        areas = widths * heights
        small_mask = areas < (self.small_object_threshold ** 2)
        return small_mask

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'cls_scores', 'bbox_preds', 'obj_scores'
            targets: list of dicts with 'bboxes' and 'labels'

        Returns:
            dict: losses
        """
        device = predictions['cls_scores'][0].device
        batch_size = predictions['cls_scores'][0].shape[0]

        total_cls_loss = 0
        total_bbox_loss = 0
        total_obj_loss = 0
        num_positives = 0

        # Process each feature level
        for level_idx in range(len(predictions['cls_scores'])):
            cls_scores = predictions['cls_scores'][level_idx]
            bbox_preds = predictions['bbox_preds'][level_idx]
            obj_scores = predictions['obj_scores'][level_idx]

            # Reshape predictions
            batch, _, h, w = cls_scores.shape
            cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
            obj_scores = obj_scores.permute(0, 2, 3, 1).reshape(-1)

            # TODO: Implement matching strategy (e.g., SimOTA, ATSS)
            # For now, this is a simplified version
            # In practice, you need to match predictions to targets

        # Placeholder loss computation
        # This should be replaced with proper target assignment
        cls_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.cls_loss_weight * cls_loss +
            self.bbox_loss_weight * bbox_loss +
            self.obj_loss_weight * obj_loss
        )

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'obj_loss': obj_loss
        }


class YOLOLoss(nn.Module):
    """
    YOLO-style loss for small object detection
    Simplified implementation
    """
    def __init__(self, num_classes):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.giou = GIoULoss()

    def forward(self, predictions, targets):
        """Compute YOLO loss"""
        # Placeholder implementation
        # Full implementation requires anchor matching
        device = predictions['cls_scores'][0].device

        loss = torch.tensor(0.0, device=device)

        return {
            'total_loss': loss,
            'cls_loss': loss,
            'bbox_loss': loss,
            'obj_loss': loss
        }


if __name__ == "__main__":
    # Test losses
    print("Testing Focal Loss...")
    focal = FocalLoss()
    inputs = torch.randn(10, 80)
    targets = torch.randint(0, 80, (10,))
    loss = focal(inputs, targets)
    print(f"Focal loss: {loss.item():.4f}")

    print("\nTesting GIoU Loss...")
    giou = GIoULoss()
    pred_boxes = torch.rand(10, 4) * 100
    target_boxes = torch.rand(10, 4) * 100
    # Ensure valid boxes
    pred_boxes[:, 2:] += pred_boxes[:, :2]
    target_boxes[:, 2:] += target_boxes[:, :2]
    loss = giou(pred_boxes, target_boxes)
    print(f"GIoU loss: {loss.item():.4f}")
