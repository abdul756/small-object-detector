"""
Dataset module for small object detection
Supports COCO, YOLO, and Pascal VOC formats
"""
import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SmallObjectDataset(Dataset):
    """
    Dataset for small object detection
    Features:
    - Multiple annotation format support (COCO, YOLO, Pascal VOC)
    - Heavy augmentation for small objects
    - Mosaic and MixUp augmentation
    """
    def __init__(
        self,
        img_dir,
        ann_file,
        img_size=(640, 640),
        augment=True,
        format='coco',
        mosaic_prob=0.5,
        mixup_prob=0.3
    ):
        self.img_dir = Path(img_dir)
        self.ann_file = ann_file
        self.img_size = img_size
        self.augment = augment
        self.format = format
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

        # Load annotations
        self.samples = self._load_annotations()

        # Build transforms
        self.transform = self._build_transforms()

    def _load_annotations(self):
        """Load annotations based on format"""
        if self.format == 'coco':
            return self._load_coco_annotations()
        elif self.format == 'yolo':
            return self._load_yolo_annotations()
        elif self.format == 'pascal_voc':
            return self._load_voc_annotations()
        else:
            raise ValueError(f"Unknown annotation format: {self.format}")

    def _load_coco_annotations(self):
        """Load COCO format annotations"""
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build image id to annotations mapping
        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        # Build samples
        samples = []
        for img_info in coco_data['images']:
            img_id = img_info['id']
            img_path = self.img_dir / img_info['file_name']

            if img_id in img_id_to_anns:
                bboxes = []
                labels = []
                for ann in img_id_to_anns[img_id]:
                    # COCO bbox: [x, y, w, h]
                    x, y, w, h = ann['bbox']
                    bboxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
                    labels.append(ann['category_id'])

                samples.append({
                    'image_path': str(img_path),
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64)
                })

        return samples

    def _load_yolo_annotations(self):
        """Load YOLO format annotations"""
        samples = []
        # YOLO format: each image has a corresponding .txt file
        # Format: class_id x_center y_center width height (normalized)
        for img_path in self.img_dir.glob('*.jpg'):
            ann_path = img_path.with_suffix('.txt')
            if not ann_path.exists():
                continue

            # Read image size
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            bboxes = []
            labels = []
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert from normalized to pixel coordinates
                    x1 = (x_center - width / 2) * w
                    y1 = (y_center - height / 2) * h
                    x2 = (x_center + width / 2) * w
                    y2 = (y_center + height / 2) * h

                    bboxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

            if len(bboxes) > 0:
                samples.append({
                    'image_path': str(img_path),
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64)
                })

        return samples

    def _load_voc_annotations(self):
        """Load Pascal VOC format annotations"""
        # Implementation for VOC format
        raise NotImplementedError("Pascal VOC format not yet implemented")

    def _build_transforms(self):
        """Build augmentation pipeline optimized for small objects"""
        if self.augment:
            return A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),

                # Color augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.5),

                # Blur and noise for robustness
                A.OneOf([
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.2),

                # Resize
                A.Resize(self.img_size[0], self.img_size[1]),

                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))

    def _apply_mosaic(self, idx):
        """
        Mosaic augmentation: combine 4 images into one
        Effective for small object detection
        """
        # Select 3 additional random images
        indices = [idx] + [np.random.randint(0, len(self)) for _ in range(3)]

        # Create mosaic canvas
        mosaic_h, mosaic_w = self.img_size[0] * 2, self.img_size[1] * 2
        mosaic_img = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

        mosaic_bboxes = []
        mosaic_labels = []

        # Place images in 2x2 grid
        for i, idx in enumerate(indices):
            sample = self.samples[idx]
            img = cv2.imread(sample['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Resize to half of target size
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

            # Determine position in mosaic
            x_offset = (i % 2) * self.img_size[1]
            y_offset = (i // 2) * self.img_size[0]

            # Place image
            mosaic_img[y_offset:y_offset + self.img_size[0],
                      x_offset:x_offset + self.img_size[1]] = img

            # Adjust bboxes
            bboxes = sample['bboxes'].copy()
            if len(bboxes) > 0:
                # Scale bboxes
                scale_x = self.img_size[1] / w
                scale_y = self.img_size[0] / h
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x + x_offset
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y + y_offset

                mosaic_bboxes.append(bboxes)
                mosaic_labels.extend(sample['labels'])

        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.vstack(mosaic_bboxes)
        else:
            mosaic_bboxes = np.array([])

        return mosaic_img, mosaic_bboxes, np.array(mosaic_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample"""
        # Apply mosaic augmentation
        if self.augment and np.random.rand() < self.mosaic_prob:
            img, bboxes, labels = self._apply_mosaic(idx)
        else:
            sample = self.samples[idx]
            img = cv2.imread(sample['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = sample['bboxes']
            labels = sample['labels']

        # Apply transforms
        if len(bboxes) > 0:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                labels=labels
            )
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            transformed = self.transform(image=img, bboxes=[], labels=[])
            img = transformed['image']
            bboxes = []
            labels = []

        # Convert to tensors
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            'image': img,
            'bboxes': bboxes,
            'labels': labels
        }


def collate_fn(batch):
    """Custom collate function for variable number of boxes"""
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]

    return {
        'images': images,
        'bboxes': bboxes,
        'labels': labels
    }


if __name__ == "__main__":
    # Test dataset
    dataset = SmallObjectDataset(
        img_dir='data/raw/train',
        ann_file='data/annotations/train.json',
        img_size=(640, 640),
        augment=True,
        format='coco'
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Number of boxes: {len(sample['bboxes'])}")
        print(f"Labels: {sample['labels']}")
