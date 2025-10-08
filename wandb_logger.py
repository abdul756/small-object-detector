"""
WandB Logger for Aerial Person Detection
Handles logging metrics, images, and predictions to Weights & Biases
"""
import wandb
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class WandbLogger:
    """
    WandB logger for person detection
    Features:
    - Log training/validation metrics
    - Log images with ground truth and predictions
    - Log model artifacts
    - Track hyperparameters
    """

    def __init__(self, config: Dict, model=None, resume: bool = False):
        """
        Initialize WandB logger

        Args:
            config: Configuration dictionary
            model: Model to watch (optional)
            resume: Whether to resume from previous run
        """
        self.config = config
        self.enabled = config['logging']['use_wandb']

        if not self.enabled:
            print("WandB logging is disabled in config")
            return

        # Extract WandB settings
        wandb_config = config['logging'].get('wandb', {})
        self.project_name = config['logging']['project_name']
        self.experiment_name = config['logging']['experiment_name']
        self.entity = wandb_config.get('entity', None)
        self.tags = wandb_config.get('tags', [])
        self.notes = wandb_config.get('notes', '')
        self.log_images = wandb_config.get('log_images', True)
        self.log_predictions = wandb_config.get('log_predictions', True)
        self.max_images = wandb_config.get('max_images_to_log', 100)

        # Initialize WandB
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            entity=self.entity,
            config=config,
            tags=self.tags,
            notes=self.notes,
            resume='allow' if resume else None
        )

        # Watch model if provided
        if model is not None:
            wandb.watch(model, log='all', log_freq=100)

        print(f"WandB initialized: {self.run.url}")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None, prefix: str = ''):
        """
        Log metrics to WandB

        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if not self.enabled:
            return

        # Add prefix to all metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        wandb.log(metrics, step=step)

    def log_images_with_boxes(
        self,
        images: Union[torch.Tensor, np.ndarray],
        gt_boxes: List[np.ndarray],
        gt_labels: List[np.ndarray],
        pred_boxes: Optional[List[np.ndarray]] = None,
        pred_labels: Optional[List[np.ndarray]] = None,
        pred_scores: Optional[List[np.ndarray]] = None,
        step: Optional[int] = None,
        split: str = 'val',
        max_images: Optional[int] = None
    ):
        """
        Log images with bounding boxes to WandB

        Args:
            images: Batch of images [B, C, H, W] or [B, H, W, C]
            gt_boxes: List of ground truth boxes for each image
            gt_labels: List of ground truth labels for each image
            pred_boxes: List of predicted boxes for each image (optional)
            pred_labels: List of predicted labels for each image (optional)
            pred_scores: List of prediction scores for each image (optional)
            step: Training step/epoch
            split: Dataset split ('train', 'val', 'test')
            max_images: Maximum number of images to log
        """
        if not self.enabled or not self.log_images:
            return

        max_images = max_images or self.max_images
        num_images = min(len(images), max_images)

        wandb_images = []

        for i in range(num_images):
            # Get image
            img = self._prepare_image(images[i])

            # Prepare boxes for WandB
            boxes_data = []

            # Add ground truth boxes
            if gt_boxes is not None and len(gt_boxes[i]) > 0:
                for box, label in zip(gt_boxes[i], gt_labels[i]):
                    boxes_data.append({
                        "position": {
                            "minX": float(box[0]),
                            "minY": float(box[1]),
                            "maxX": float(box[2]),
                            "maxY": float(box[3])
                        },
                        "class_id": int(label),
                        "box_caption": "GT: person",
                        "domain": "pixel"
                    })

            # Add prediction boxes
            if pred_boxes is not None and len(pred_boxes[i]) > 0:
                for j, (box, label) in enumerate(zip(pred_boxes[i], pred_labels[i])):
                    score = pred_scores[i][j] if pred_scores is not None else 0.0
                    boxes_data.append({
                        "position": {
                            "minX": float(box[0]),
                            "minY": float(box[1]),
                            "maxX": float(box[2]),
                            "maxY": float(box[3])
                        },
                        "class_id": int(label) + 10,  # Offset to distinguish from GT
                        "box_caption": f"Pred: {score:.2f}",
                        "scores": {"confidence": float(score)},
                        "domain": "pixel"
                    })

            # Create WandB image with boxes
            wandb_img = wandb.Image(
                img,
                boxes={
                    "predictions": {
                        "box_data": boxes_data,
                        "class_labels": {
                            0: "GT_person",
                            10: "Pred_person"
                        }
                    }
                }
            )
            wandb_images.append(wandb_img)

        # Log images
        wandb.log({f"{split}/predictions": wandb_images}, step=step)

    def log_prediction_grid(
        self,
        images: Union[torch.Tensor, np.ndarray],
        pred_boxes: List[np.ndarray],
        pred_labels: List[np.ndarray],
        pred_scores: List[np.ndarray],
        gt_boxes: Optional[List[np.ndarray]] = None,
        gt_labels: Optional[List[np.ndarray]] = None,
        step: Optional[int] = None,
        split: str = 'test',
        num_images: int = 16
    ):
        """
        Log a grid of images with predictions visualized

        Args:
            images: Batch of images
            pred_boxes: List of predicted boxes
            pred_labels: List of predicted labels
            pred_scores: List of prediction scores
            gt_boxes: List of ground truth boxes (optional)
            gt_labels: List of ground truth labels (optional)
            step: Training step/epoch
            split: Dataset split
            num_images: Number of images to include in grid
        """
        if not self.enabled or not self.log_predictions:
            return

        num_images = min(len(images), num_images)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        for idx in range(num_images):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # Prepare image
            img = self._prepare_image(images[idx])
            ax.imshow(img)

            # Draw ground truth boxes (green)
            if gt_boxes is not None and len(gt_boxes[idx]) > 0:
                for box in gt_boxes[idx]:
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none',
                        label='GT'
                    )
                    ax.add_patch(rect)

            # Draw prediction boxes (red) with scores
            num_preds = 0
            if pred_boxes is not None and len(pred_boxes[idx]) > 0:
                for box, score in zip(pred_boxes[idx], pred_scores[idx]):
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add score text
                    ax.text(
                        box[0], box[1] - 5,
                        f'{score:.2f}',
                        color='red',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
                    num_preds += 1

            gt_count = len(gt_boxes[idx]) if gt_boxes is not None and len(gt_boxes[idx]) > 0 else 0
            ax.set_title(f'GT: {gt_count} | Pred: {num_preds}', fontsize=10)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        # Log to WandB
        wandb.log({f"{split}/prediction_grid": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        step: Optional[int] = None
    ):
        """Log confusion matrix to WandB"""
        if not self.enabled:
            return

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        }, step=step)

    def log_histogram(self, data: Dict[str, np.ndarray], step: Optional[int] = None):
        """Log histogram to WandB"""
        if not self.enabled:
            return

        for name, values in data.items():
            wandb.log({name: wandb.Histogram(values)}, step=step)

    def log_model_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None):
        """
        Log model checkpoint as artifact

        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Optional metadata dictionary
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name=f"model-{self.experiment_name}",
            type="model",
            metadata=metadata or {}
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    def _prepare_image(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Prepare image for logging (denormalize if needed)

        Args:
            img: Image tensor or array

        Returns:
            RGB image as numpy array [H, W, 3] in range [0, 255]
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        # Handle different formats
        if img.ndim == 3:
            if img.shape[0] == 3:  # [C, H, W]
                img = img.transpose(1, 2, 0)  # [H, W, C]

        # Denormalize if needed (assuming ImageNet normalization)
        if img.max() <= 1.5:  # Likely normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean

        # Clip and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

    def finish(self):
        """Finish WandB run"""
        if self.enabled and self.run is not None:
            wandb.finish()
            print("WandB run finished")


def test_wandb_logger():
    """Test WandB logger functionality"""
    # Create dummy config
    config = {
        'logging': {
            'project_name': 'aerial-person-detection-test',
            'experiment_name': 'test_run',
            'use_wandb': True,
            'wandb': {
                'entity': None,
                'tags': ['test'],
                'notes': 'Testing WandB logger',
                'log_images': True,
                'log_predictions': True,
                'max_images_to_log': 4
            }
        },
        'data': {'img_size': [1024, 1024], 'num_classes': 1}
    }

    # Initialize logger
    logger = WandbLogger(config)

    # Create dummy data
    images = torch.randn(4, 3, 1024, 1024)
    gt_boxes = [
        np.array([[100, 100, 150, 150], [200, 200, 250, 250]]),
        np.array([[300, 300, 350, 350]]),
        np.array([[400, 400, 450, 450], [500, 500, 550, 550], [600, 600, 650, 650]]),
        np.array([[50, 50, 100, 100]])
    ]
    gt_labels = [
        np.array([0, 0]),
        np.array([0]),
        np.array([0, 0, 0]),
        np.array([0])
    ]
    pred_boxes = [
        np.array([[105, 105, 155, 155], [205, 205, 255, 255]]),
        np.array([[305, 305, 355, 355]]),
        np.array([[405, 405, 455, 455], [505, 505, 555, 555]]),
        np.array([[55, 55, 105, 105]])
    ]
    pred_labels = [
        np.array([0, 0]),
        np.array([0]),
        np.array([0, 0]),
        np.array([0])
    ]
    pred_scores = [
        np.array([0.95, 0.87]),
        np.array([0.92]),
        np.array([0.88, 0.76]),
        np.array([0.91])
    ]

    # Log metrics
    logger.log_metrics({'loss': 0.5, 'acc': 0.9}, step=1, prefix='train/')

    # Log images with boxes
    logger.log_images_with_boxes(
        images, gt_boxes, gt_labels,
        pred_boxes, pred_labels, pred_scores,
        step=1, split='test'
    )

    # Log prediction grid
    logger.log_prediction_grid(
        images, pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels,
        step=1, split='test', num_images=4
    )

    # Finish
    logger.finish()


if __name__ == "__main__":
    test_wandb_logger()
