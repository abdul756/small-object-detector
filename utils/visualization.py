"""
Visualization utilities for object detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def generate_colors(num_classes):
    """Generate distinct colors for each class"""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
    return colors


def draw_boxes(image, boxes, labels, scores=None, class_names=None, colors=None, thickness=2):
    """
    Draw bounding boxes on image
    Args:
        image: numpy array (H, W, 3)
        boxes: numpy array (N, 4) [x1, y1, x2, y2]
        labels: numpy array (N,) class labels
        scores: numpy array (N,) confidence scores
        class_names: list of class names
        colors: list of colors for each class
        thickness: line thickness
    Returns:
        image with boxes drawn
    """
    image = image.copy()

    if class_names is None:
        class_names = COCO_CLASSES

    if colors is None:
        colors = generate_colors(len(class_names))

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)

        # Get color for this class
        color = colors[int(label) % len(colors)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        if int(label) < len(class_names):
            class_name = class_names[int(label)]
        else:
            class_name = f"Class {int(label)}"

        if scores is not None:
            label_text = f"{class_name}: {scores[i]:.2f}"
        else:
            label_text = class_name

        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return image


def visualize_predictions(
    image,
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes=None,
    gt_labels=None,
    class_names=None,
    save_path=None,
    show=True
):
    """
    Visualize predictions and optionally ground truth
    Args:
        image: numpy array or path to image
        pred_boxes: predicted boxes (N, 4)
        pred_labels: predicted labels (N,)
        pred_scores: prediction scores (N,)
        gt_boxes: ground truth boxes (M, 4)
        gt_labels: ground truth labels (M,)
        class_names: list of class names
        save_path: path to save visualization
        show: whether to display the image
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2 if gt_boxes is not None else 1, figsize=(15, 8))
    if gt_boxes is None:
        axes = [axes]

    # Draw predictions
    pred_img = draw_boxes(
        image,
        pred_boxes,
        pred_labels,
        pred_scores,
        class_names
    )
    axes[0].imshow(pred_img)
    axes[0].set_title(f"Predictions ({len(pred_boxes)} detections)")
    axes[0].axis('off')

    # Draw ground truth
    if gt_boxes is not None:
        gt_img = draw_boxes(
            image,
            gt_boxes,
            gt_labels,
            class_names=class_names
        )
        axes[1].imshow(gt_img)
        axes[1].set_title(f"Ground Truth ({len(gt_boxes)} objects)")
        axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file
    Args:
        log_file: path to training log
        save_path: path to save plot
    """
    # This would parse the log file and plot training curves
    # Implementation depends on log format
    pass


def visualize_small_objects(image, boxes, labels, scores, size_threshold=32):
    """
    Highlight small objects in the image
    Args:
        image: input image
        boxes: bounding boxes
        labels: class labels
        scores: confidence scores
        size_threshold: threshold for small objects (pixels)
    """
    image = image.copy()

    # Separate small and large objects
    small_boxes = []
    small_labels = []
    small_scores = []

    large_boxes = []
    large_labels = []
    large_scores = []

    for box, label, score in zip(boxes, labels, scores):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height

        if area < (size_threshold ** 2):
            small_boxes.append(box)
            small_labels.append(label)
            small_scores.append(score)
        else:
            large_boxes.append(box)
            large_labels.append(label)
            large_scores.append(score)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Draw all objects
    all_img = draw_boxes(
        image,
        boxes,
        labels,
        scores
    )
    axes[0].imshow(all_img)
    axes[0].set_title(f"All Objects ({len(boxes)} total)")
    axes[0].axis('off')

    # Draw only small objects
    if len(small_boxes) > 0:
        small_img = draw_boxes(
            image,
            np.array(small_boxes),
            np.array(small_labels),
            np.array(small_scores),
            thickness=3  # Thicker for visibility
        )
        axes[1].imshow(small_img)
        axes[1].set_title(f"Small Objects ({len(small_boxes)} detected)")
        axes[1].axis('off')
    else:
        axes[1].imshow(image)
        axes[1].set_title("No Small Objects Detected")
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...")

    # Create dummy image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255

    # Dummy predictions
    pred_boxes = np.array([
        [50, 50, 150, 150],
        [200, 200, 300, 350],
        [400, 100, 450, 140]
    ])
    pred_labels = np.array([0, 16, 2])
    pred_scores = np.array([0.95, 0.87, 0.76])

    # Draw boxes
    result = draw_boxes(image, pred_boxes, pred_labels, pred_scores)
    print(f"Result image shape: {result.shape}")

    # Test small object visualization
    visualize_small_objects(image, pred_boxes, pred_labels, pred_scores)
