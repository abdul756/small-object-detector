"""
Test script for the aerial person detection dataloader
Demonstrates loading and visualizing data from all three annotation formats
"""
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataloader import build_dataloader


def denormalize_image(img_tensor):
    """Denormalize image for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def visualize_batch(batch, num_images=4, title="Aerial Person Detection"):
    """Visualize images with bounding boxes"""
    images = batch['images']
    bboxes_list = batch['bboxes']
    labels_list = batch['labels']

    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]

        # Get image
        img = denormalize_image(images[idx])
        ax.imshow(img)

        # Get bboxes for this image
        bboxes = bboxes_list[idx].cpu().numpy()
        labels = labels_list[idx].cpu().numpy()

        # Draw bounding boxes
        h, w = img.shape[:2]
        for bbox, label in zip(bboxes, labels):
            # Bboxes are in normalized coordinates after transforms
            x1, y1, x2, y2 = bbox

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f'Image {idx + 1}: {len(bboxes)} people detected')
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def test_dataloader(format_name='yolo'):
    """Test dataloader with specified format"""
    print(f"\n{'='*60}")
    print(f"Testing {format_name.upper()} format")
    print(f"{'='*60}")

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set format
    config['data']['annotations_format'] = format_name

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader = build_dataloader(config, split='train')
    val_loader = build_dataloader(config, split='val')
    test_loader = build_dataloader(config, split='test')

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Valid: {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    print(f"  Total: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} samples")

    # Get a batch
    print("\nFetching a batch from training set...")
    batch = next(iter(train_loader))

    print(f"\nBatch Information:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Batch size: {len(batch['bboxes'])}")
    print(f"  Number of people in first image: {len(batch['bboxes'][0])}")
    print(f"  Number of people in second image: {len(batch['bboxes'][1])}")

    # Calculate average objects per image in batch
    avg_objects = sum(len(b) for b in batch['bboxes']) / len(batch['bboxes'])
    print(f"  Average people per image: {avg_objects:.2f}")

    return batch


def main():
    """Main test function"""
    print("="*60)
    print("Aerial Person Detection Dataloader Test")
    print("="*60)

    # Test all formats
    formats = ['yolo', 'pascal_voc', 'coco']

    for fmt in formats:
        try:
            batch = test_dataloader(fmt)
            print(f"\n✓ {fmt.upper()} format: SUCCESS")
        except Exception as e:
            print(f"\n✗ {fmt.upper()} format: FAILED - {e}")

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)

    # Optional: Visualize a batch (requires matplotlib)
    try:
        print("\nGenerating visualization...")
        # Load config and build dataloader for visualization
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Use YOLO format for visualization
        config['data']['annotations_format'] = 'yolo'
        # Disable augmentation for clearer visualization
        loader = build_dataloader(config, split='val')

        batch = next(iter(loader))
        fig = visualize_batch(batch, num_images=4, title="Aerial Person Detection - Sample Images")

        # Save figure
        output_path = 'dataloader_visualization.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
