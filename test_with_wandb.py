"""
Test inference script with WandB logging
Demonstrates how to run inference and visualize predictions with scores in WandB
"""
import yaml
import torch
import numpy as np
from pathlib import Path
from dataloader import build_dataloader
from wandb_logger import WandbLogger
import argparse


class DummyDetector:
    """
    Dummy detector for testing WandB visualization
    In production, replace this with your actual trained model
    """

    def __init__(self, conf_threshold=0.25, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def predict(self, images, gt_boxes=None):
        """
        Generate dummy predictions
        In production, replace with actual model inference

        Args:
            images: Batch of images [B, C, H, W]
            gt_boxes: Ground truth boxes (used here to create realistic dummy predictions)

        Returns:
            pred_boxes, pred_labels, pred_scores for each image in batch
        """
        batch_size = images.shape[0]
        pred_boxes_list = []
        pred_labels_list = []
        pred_scores_list = []

        for i in range(batch_size):
            if gt_boxes is not None and len(gt_boxes[i]) > 0:
                # Create predictions based on GT (with some noise for testing)
                num_gt = len(gt_boxes[i])

                # Keep ~80% of GT boxes as predictions
                keep_indices = np.random.choice(
                    num_gt,
                    size=max(1, int(num_gt * 0.8)),
                    replace=False
                )

                pred_boxes = gt_boxes[i][keep_indices].copy()
                # Add some noise to boxes
                noise = np.random.randn(*pred_boxes.shape) * 5
                pred_boxes = pred_boxes + noise

                # Generate random scores
                pred_scores = np.random.uniform(
                    self.conf_threshold, 1.0,
                    size=len(pred_boxes)
                )

                # Add some false positives
                num_fp = np.random.randint(0, 3)
                if num_fp > 0:
                    h, w = images.shape[2], images.shape[3]
                    fp_boxes = np.random.rand(num_fp, 4)
                    fp_boxes[:, [0, 2]] *= w
                    fp_boxes[:, [1, 3]] *= h
                    fp_boxes[:, 2:] += fp_boxes[:, :2]  # Convert to x2, y2
                    fp_scores = np.random.uniform(
                        self.conf_threshold, 0.7,
                        size=num_fp
                    )

                    pred_boxes = np.vstack([pred_boxes, fp_boxes])
                    pred_scores = np.concatenate([pred_scores, fp_scores])

                # Sort by scores
                sort_indices = np.argsort(pred_scores)[::-1]
                pred_boxes = pred_boxes[sort_indices]
                pred_scores = pred_scores[sort_indices]

                pred_labels = np.zeros(len(pred_boxes), dtype=np.int64)

            else:
                # No GT, create random predictions
                num_preds = np.random.randint(0, 5)
                h, w = images.shape[2], images.shape[3]

                pred_boxes = np.random.rand(num_preds, 4)
                pred_boxes[:, [0, 2]] *= w
                pred_boxes[:, [1, 3]] *= h
                pred_boxes[:, 2:] += pred_boxes[:, :2]

                pred_scores = np.random.uniform(
                    self.conf_threshold, 1.0,
                    size=num_preds
                )
                pred_labels = np.zeros(num_preds, dtype=np.int64)

            pred_boxes_list.append(pred_boxes.astype(np.float32))
            pred_labels_list.append(pred_labels)
            pred_scores_list.append(pred_scores.astype(np.float32))

        return pred_boxes_list, pred_labels_list, pred_scores_list


def run_inference_with_wandb(
    config_path: str = 'configs/config.yaml',
    num_batches: int = 5,
    use_dummy_model: bool = True,
    model_path: str = None
):
    """
    Run inference on test set and log predictions to WandB

    Args:
        config_path: Path to configuration file
        num_batches: Number of batches to process
        use_dummy_model: Whether to use dummy model (for testing)
        model_path: Path to trained model checkpoint (if use_dummy_model=False)
    """
    print("="*60)
    print("Running Inference with WandB Logging")
    print("="*60)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize WandB logger
    print("\nInitializing WandB...")
    logger = WandbLogger(config)

    # Load model
    if use_dummy_model:
        print("\nUsing dummy detector for testing...")
        model = DummyDetector(
            conf_threshold=config['inference']['conf_threshold'],
            nms_threshold=config['inference']['iou_threshold']
        )
    else:
        print(f"\nLoading model from {model_path}...")
        # TODO: Load your actual trained model here
        # model = YourModel.load_from_checkpoint(model_path)
        # model.eval()
        raise NotImplementedError("Actual model loading not implemented yet")

    # Build test dataloader
    print("\nBuilding test dataloader...")
    test_loader = build_dataloader(config, split='test')
    print(f"Test dataset: {len(test_loader.dataset)} samples")

    # Run inference
    print(f"\nRunning inference on {num_batches} batches...")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    all_images = []
    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break

            images = batch['images'].to(device)
            gt_boxes = batch['bboxes']
            gt_labels = batch['labels']

            print(f"\nBatch {batch_idx + 1}/{num_batches}:")
            print(f"  Images: {images.shape}")
            print(f"  GT boxes: {[len(b) for b in gt_boxes]}")

            # Run inference
            if use_dummy_model:
                # Convert GT boxes to numpy for dummy model
                gt_boxes_np = [b.cpu().numpy() for b in gt_boxes]
                pred_boxes, pred_labels, pred_scores = model.predict(images, gt_boxes_np)
            else:
                # TODO: Actual model inference
                pred_boxes, pred_labels, pred_scores = model(images)

            print(f"  Predictions: {[len(b) for b in pred_boxes]}")
            print(f"  Avg confidence: {[np.mean(s) if len(s) > 0 else 0 for s in pred_scores]}")

            # Collect for batch logging
            all_images.extend([img.cpu() for img in images])
            all_gt_boxes.extend([b.cpu().numpy() if isinstance(b, torch.Tensor) else b for b in gt_boxes])
            all_gt_labels.extend([l.cpu().numpy() if isinstance(l, torch.Tensor) else l for l in gt_labels])
            all_pred_boxes.extend(pred_boxes)
            all_pred_labels.extend(pred_labels)
            all_pred_scores.extend(pred_scores)

    # Log predictions to WandB
    print(f"\n\nLogging {len(all_images)} images to WandB...")

    # Log with bounding boxes
    logger.log_images_with_boxes(
        images=all_images,
        gt_boxes=all_gt_boxes,
        gt_labels=all_gt_labels,
        pred_boxes=all_pred_boxes,
        pred_labels=all_pred_labels,
        pred_scores=all_pred_scores,
        step=0,
        split='test',
        max_images=50
    )

    # Log prediction grid
    logger.log_prediction_grid(
        images=all_images[:16],
        pred_boxes=all_pred_boxes[:16],
        pred_labels=all_pred_labels[:16],
        pred_scores=all_pred_scores[:16],
        gt_boxes=all_gt_boxes[:16],
        gt_labels=all_gt_labels[:16],
        step=0,
        split='test',
        num_images=16
    )

    # Calculate and log statistics
    total_gt = sum(len(b) for b in all_gt_boxes)
    total_pred = sum(len(b) for b in all_pred_boxes)
    avg_conf = np.mean([np.mean(s) if len(s) > 0 else 0 for s in all_pred_scores])

    stats = {
        'test/total_images': len(all_images),
        'test/total_gt_boxes': total_gt,
        'test/total_predictions': total_pred,
        'test/avg_gt_per_image': total_gt / len(all_images),
        'test/avg_pred_per_image': total_pred / len(all_images),
        'test/avg_confidence': avg_conf
    }

    logger.log_metrics(stats, step=0)

    print("\n\nInference Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    # Finish WandB run
    print(f"\n\nWandB run URL: {logger.run.url}")
    print("Check WandB for visualizations!")

    logger.finish()

    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Run inference with WandB logging')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=5,
        help='Number of batches to process'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--use-real-model',
        action='store_true',
        help='Use real model instead of dummy detector'
    )

    args = parser.parse_args()

    run_inference_with_wandb(
        config_path=args.config,
        num_batches=args.num_batches,
        use_dummy_model=not args.use_real_model,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()
