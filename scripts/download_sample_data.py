"""
Script to download sample COCO dataset for testing
"""
import os
import urllib.request
import json
from pathlib import Path
from tqdm import tqdm


def download_file(url, filename):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)


def download_coco_sample():
    """Download a sample of COCO dataset"""
    print("Downloading COCO sample dataset...")

    # Create directories
    base_dir = Path('data')
    (base_dir / 'raw' / 'train').mkdir(parents=True, exist_ok=True)
    (base_dir / 'raw' / 'val').mkdir(parents=True, exist_ok=True)
    (base_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    # Download annotations
    print("\nDownloading annotations...")
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_file = base_dir / "annotations.zip"

    # Note: This is a large file (241MB)
    # For testing, you may want to create a smaller custom dataset

    print("\nTo get COCO dataset:")
    print("1. Download from: https://cocodataset.org/#download")
    print("2. Or create your own custom dataset")
    print("3. Place images in data/raw/train and data/raw/val")
    print("4. Place annotations in data/annotations/")

    # Create a dummy annotation file for structure reference
    dummy_coco = {
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"}
        ]
    }

    # Save dummy annotation
    with open(base_dir / 'annotations' / 'example_format.json', 'w') as f:
        json.dump(dummy_coco, f, indent=2)

    print("\nCreated example annotation format in data/annotations/example_format.json")


if __name__ == "__main__":
    download_coco_sample()
