"""
Metrics for object detection evaluation
Includes mAP, precision, recall, etc.
"""
import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes
    Args:
        box1: (4,) [x1, y1, x2, y2]
        box2: (4,) [x1, y1, x2, y2]
    Returns:
        float: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_ap(recalls, precisions):
    """
    Compute Average Precision using 11-point interpolation
    Args:
        recalls: list of recall values
        precisions: list of precision values
    Returns:
        float: Average Precision
    """
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap


class MeanAveragePrecision:
    """
    Mean Average Precision (mAP) calculator
    Supports multiple IoU thresholds
    """
    def __init__(self, num_classes, iou_thresholds=[0.5, 0.75]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        """Reset internal state"""
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)

    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        """
        Update with a batch of predictions and ground truths
        Args:
            pred_boxes: (N, 4) predicted boxes
            pred_labels: (N,) predicted labels
            pred_scores: (N,) prediction scores
            gt_boxes: (M, 4) ground truth boxes
            gt_labels: (M,) ground truth labels
        """
        # Convert to numpy
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()

        # Store predictions
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            self.predictions[label].append({
                'box': box,
                'score': score
            })

        # Store ground truths
        for box, label in zip(gt_boxes, gt_labels):
            self.ground_truths[label].append({
                'box': box,
                'matched': False
            })

    def compute(self):
        """
        Compute mAP across all classes and IoU thresholds
        Returns:
            dict: mAP metrics
        """
        aps = defaultdict(dict)

        for iou_thresh in self.iou_thresholds:
            class_aps = []

            for class_id in range(self.num_classes):
                # Get predictions and ground truths for this class
                preds = self.predictions.get(class_id, [])
                gts = self.ground_truths.get(class_id, [])

                if len(gts) == 0:
                    continue

                if len(preds) == 0:
                    class_aps.append(0.0)
                    continue

                # Sort predictions by score
                preds = sorted(preds, key=lambda x: x['score'], reverse=True)

                # Compute TP and FP
                tp = np.zeros(len(preds))
                fp = np.zeros(len(preds))

                for i, pred in enumerate(preds):
                    max_iou = 0.0
                    max_idx = -1

                    # Find best matching ground truth
                    for j, gt in enumerate(gts):
                        if gt['matched']:
                            continue

                        iou = compute_iou(pred['box'], gt['box'])
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = j

                    # Check if match is valid
                    if max_iou >= iou_thresh and max_idx >= 0:
                        tp[i] = 1
                        gts[max_idx]['matched'] = True
                    else:
                        fp[i] = 1

                # Compute cumulative TP and FP
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                # Compute precision and recall
                recalls = tp_cumsum / len(gts)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

                # Compute AP
                ap = compute_ap(recalls.tolist(), precisions.tolist())
                class_aps.append(ap)

                # Reset matched flags for next IoU threshold
                for gt in gts:
                    gt['matched'] = False

            # Compute mean AP
            if len(class_aps) > 0:
                map_value = np.mean(class_aps)
                aps[f'mAP@{iou_thresh}'] = map_value
            else:
                aps[f'mAP@{iou_thresh}'] = 0.0

        # Compute mAP@[0.5:0.95]
        if len(self.iou_thresholds) > 1:
            aps['mAP'] = np.mean([aps[f'mAP@{t}'] for t in self.iou_thresholds])

        return aps


def non_max_suppression(boxes, scores, iou_threshold=0.45, score_threshold=0.25):
    """
    Non-Maximum Suppression
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold to filter boxes
    Returns:
        indices: Indices of boxes to keep
    """
    # Filter by score
    keep_mask = scores > score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if len(boxes) == 0:
        return np.array([])

    # Sort by score
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])

        # Keep boxes with IoU below threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


if __name__ == "__main__":
    # Test mAP computation
    print("Testing mAP computation...")

    metric = MeanAveragePrecision(num_classes=10)

    # Dummy predictions and ground truths
    pred_boxes = np.array([
        [10, 10, 50, 50],
        [60, 60, 100, 100],
        [15, 15, 55, 55]
    ])
    pred_labels = np.array([0, 1, 0])
    pred_scores = np.array([0.9, 0.8, 0.7])

    gt_boxes = np.array([
        [12, 12, 52, 52],
        [65, 65, 105, 105]
    ])
    gt_labels = np.array([0, 1])

    metric.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    results = metric.compute()
    print("mAP results:", results)

    # Test NMS
    print("\nTesting NMS...")
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],
        [100, 100, 150, 150]
    ])
    scores = np.array([0.9, 0.8, 0.95])

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(f"Kept boxes: {keep}")
