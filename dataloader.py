"""
DataLoader module for efficient data loading
"""
import torch
from torch.utils.data import DataLoader
from dataset import SmallObjectDataset, collate_fn


def build_dataloader(config, split='train'):
    """
    Build dataloader for training/validation/testing

    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'

    Returns:
        DataLoader instance
    """
    # Get annotation format
    ann_format = config['data']['annotations_format'].lower()

    # Get split-specific settings
    if split == 'train':
        img_dir = config['data']['train_path']
        augment = True
        shuffle = True
        batch_size = config['training']['batch_size']

        # Get annotation file/directory based on format
        if ann_format == 'coco':
            ann_file = config['data']['train_coco_json']
            ann_dir = None
        elif ann_format == 'yolo':
            ann_file = None
            ann_dir = config['data']['train_ann_dir']
        elif ann_format == 'pascal_voc':
            ann_file = None
            ann_dir = config['data']['train_voc_dir']
        else:
            raise ValueError(f"Unknown annotation format: {ann_format}")

    elif split == 'val':
        img_dir = config['data']['val_path']
        augment = False
        shuffle = False
        batch_size = config['training']['batch_size']

        # Get annotation file/directory based on format
        if ann_format == 'coco':
            ann_file = config['data']['val_coco_json']
            ann_dir = None
        elif ann_format == 'yolo':
            ann_file = None
            ann_dir = config['data']['val_ann_dir']
        elif ann_format == 'pascal_voc':
            ann_file = None
            ann_dir = config['data']['val_voc_dir']
        else:
            raise ValueError(f"Unknown annotation format: {ann_format}")

    elif split == 'test':
        img_dir = config['data']['test_path']
        augment = False
        shuffle = False
        batch_size = 1  # Test one at a time

        # Get annotation file/directory based on format
        if ann_format == 'coco':
            ann_file = config['data']['test_coco_json']
            ann_dir = None
        elif ann_format == 'yolo':
            ann_file = None
            ann_dir = config['data']['test_ann_dir']
        elif ann_format == 'pascal_voc':
            ann_file = None
            ann_dir = config['data']['test_voc_dir']
        else:
            raise ValueError(f"Unknown annotation format: {ann_format}")

    else:
        raise ValueError(f"Unknown split: {split}")

    # Build dataset
    dataset = SmallObjectDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        ann_dir=ann_dir,
        img_size=tuple(config['data']['img_size']),
        augment=augment,
        format=ann_format,
        mosaic_prob=config['data']['augmentation']['mosaic'] if augment else 0,
        mixup_prob=config['data']['augmentation']['mixup'] if augment else 0
    )

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


if __name__ == "__main__":
    import yaml

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Build train dataloader
    train_loader = build_dataloader(config, split='train')
    print(f"Train dataloader: {len(train_loader)} batches")

    # Test iteration
    for batch in train_loader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Number of samples in batch: {len(batch['bboxes'])}")
        break
