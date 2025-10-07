"""
Inference script for small object detection
Supports single image, batch, and SAHI inference
"""
import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from models.detector import build_detector
from utils.visualization import visualize_predictions, COCO_CLASSES
from utils.metrics import non_max_suppression


class Detector:
    """Inference wrapper for small object detection"""

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Build model
        print("Loading model...")
        self.model = build_detector(self.config)
        self.model = self.model.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Model loaded successfully!")

        # Inference config
        self.conf_threshold = self.config['inference']['conf_threshold']
        self.iou_threshold = self.config['inference']['iou_threshold']
        self.img_size = self.config['data']['img_size']

    def preprocess(self, image):
        """
        Preprocess image for inference
        Args:
            image: numpy array (H, W, 3) BGR
        Returns:
            tensor: (1, 3, H, W)
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        original_shape = image.shape[:2]
        image = cv2.resize(image, tuple(self.img_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image.to(self.device), original_shape

    def postprocess(self, predictions, original_shape):
        """
        Post-process model predictions
        Args:
            predictions: model outputs
            original_shape: original image shape (H, W)
        Returns:
            dict: {boxes, labels, scores}
        """
        # This is a simplified implementation
        # Real implementation needs proper decoding of predictions

        # For now, return empty predictions
        return {
            'boxes': np.array([]),
            'labels': np.array([]),
            'scores': np.array([])
        }

    @torch.no_grad()
    def predict(self, image_path):
        """
        Run inference on a single image
        Args:
            image_path: path to image
        Returns:
            dict: {boxes, labels, scores}
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_shape = image.shape[:2]

        # Preprocess
        input_tensor, _ = self.preprocess(image)

        # Forward pass
        predictions = self.model(input_tensor)

        # Postprocess
        results = self.postprocess(predictions, original_shape)

        return results

    @torch.no_grad()
    def predict_sahi(self, image_path):
        """
        Run SAHI (Slicing Aided Hyper Inference) for small objects
        Args:
            image_path: path to image
        Returns:
            dict: {boxes, labels, scores}
        """
        if not self.config['sahi']['enabled']:
            return self.predict(image_path)

        # Load image
        image = cv2.imread(str(image_path))
        original_h, original_w = image.shape[:2]

        # SAHI parameters
        slice_h = self.config['sahi']['slice_height']
        slice_w = self.config['sahi']['slice_width']
        overlap_h = self.config['sahi']['overlap_height_ratio']
        overlap_w = self.config['sahi']['overlap_width_ratio']

        stride_h = int(slice_h * (1 - overlap_h))
        stride_w = int(slice_w * (1 - overlap_w))

        all_boxes = []
        all_labels = []
        all_scores = []

        # Slide window over image
        for y in range(0, original_h, stride_h):
            for x in range(0, original_w, stride_w):
                # Extract slice
                x_end = min(x + slice_w, original_w)
                y_end = min(y + slice_h, original_h)

                slice_img = image[y:y_end, x:x_end]

                # Run detection on slice
                input_tensor, _ = self.preprocess(slice_img)
                predictions = self.model(input_tensor)
                results = self.postprocess(predictions, slice_img.shape[:2])

                # Adjust coordinates to global image
                if len(results['boxes']) > 0:
                    results['boxes'][:, [0, 2]] += x
                    results['boxes'][:, [1, 3]] += y

                    all_boxes.append(results['boxes'])
                    all_labels.append(results['labels'])
                    all_scores.append(results['scores'])

        # Combine results
        if len(all_boxes) > 0:
            all_boxes = np.vstack(all_boxes)
            all_labels = np.concatenate(all_labels)
            all_scores = np.concatenate(all_scores)

            # Apply NMS to remove duplicates
            keep = non_max_suppression(
                all_boxes,
                all_scores,
                iou_threshold=self.iou_threshold,
                score_threshold=self.conf_threshold
            )

            return {
                'boxes': all_boxes[keep],
                'labels': all_labels[keep],
                'scores': all_scores[keep]
            }
        else:
            return {
                'boxes': np.array([]),
                'labels': np.array([]),
                'scores': np.array([])
            }

    def predict_batch(self, image_dir, output_dir=None, use_sahi=False):
        """
        Run inference on a directory of images
        Args:
            image_dir: directory containing images
            output_dir: directory to save visualizations
            use_sahi: whether to use SAHI inference
        """
        image_dir = Path(image_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))

        print(f"Found {len(image_files)} images")

        # Process each image
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            # Run inference
            if use_sahi:
                result = self.predict_sahi(image_path)
            else:
                result = self.predict(image_path)

            results.append({
                'image_path': str(image_path),
                'boxes': result['boxes'],
                'labels': result['labels'],
                'scores': result['scores']
            })

            # Visualize and save
            if output_dir:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                save_path = output_dir / f"{image_path.stem}_result.jpg"
                visualize_predictions(
                    image,
                    result['boxes'],
                    result['labels'],
                    result['scores'],
                    save_path=save_path,
                    show=False
                )

        return results


def main():
    parser = argparse.ArgumentParser(description='Small Object Detection Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='inference/results',
                        help='Output directory for results')
    parser.add_argument('--sahi', action='store_true',
                        help='Use SAHI for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create detector
    detector = Detector(args.config, args.checkpoint, args.device)

    # Run inference
    if args.image:
        print(f"Running inference on {args.image}")
        if args.sahi:
            results = detector.predict_sahi(args.image)
        else:
            results = detector.predict(args.image)

        print(f"Detected {len(results['boxes'])} objects")

        # Visualize
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visualize_predictions(
            image,
            results['boxes'],
            results['labels'],
            results['scores'],
            save_path=output_path,
            show=True
        )

    elif args.image_dir:
        print(f"Running inference on directory {args.image_dir}")
        detector.predict_batch(
            args.image_dir,
            output_dir=args.output_dir,
            use_sahi=args.sahi
        )

    else:
        print("Please specify either --image or --image_dir")


if __name__ == "__main__":
    main()
