# Small Object Detector

A state-of-the-art PyTorch implementation for detecting small objects in images. This project showcases expertise in computer vision with modern deep learning techniques optimized for small object detection.

## 🚀 Features

- **Modern Architecture**: ResNet50 backbone + FPN/PANet neck + Decoupled detection head
- **Small Object Optimization**:
  - Feature Pyramid Network (FPN) for multi-scale detection
  - CBAM attention mechanism
  - GIoU loss for better small object localization
  - Focal loss for handling class imbalance
  - SAHI (Slicing Aided Hyper Inference) support
- **Advanced Augmentation**: Mosaic, MixUp, color jittering, geometric transforms
- **Flexible**: Supports COCO, YOLO, and Pascal VOC annotation formats
- **Production Ready**: Includes training, validation, and inference pipelines

## 📁 Project Structure

```
small-object-detector/
├── configs/                    # Configuration files
│   └── config.yaml            # Main config
├── models/                    # Model architectures
│   ├── backbones/            # Backbone networks (ResNet, etc.)
│   ├── necks/                # Feature fusion (FPN, PANet)
│   ├── heads/                # Detection heads
│   ├── attention.py          # Attention modules (CBAM, SE)
│   └── detector.py           # Main detector model
├── utils/                     # Utility functions
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics (mAP)
│   └── visualization.py      # Visualization tools
├── data/                      # Dataset directory
│   ├── raw/                  # Raw images
│   ├── processed/            # Processed data
│   └── annotations/          # Annotation files
├── dataset.py                 # Dataset loader
├── dataloader.py             # DataLoader builder
├── train.py                  # Training script
├── inference.py              # Inference script
├── notebooks/                # Jupyter notebooks
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── requirements.txt          # Dependencies

```

## 🛠️ Installation

1. Clone the repository:
```bash
cd small-object-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### COCO Format
Place your data in the following structure:
```
data/
├── raw/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image1.jpg
│       └── image2.jpg
└── annotations/
    ├── train.json
    └── val.json
```

### YOLO Format
```
data/
├── raw/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image1.txt
│   │   ├── image2.jpg
│   │   └── image2.txt
│   └── val/
```

## 🏋️ Training

1. Configure your training in `configs/config.yaml`

2. Start training:
```bash
python train.py --config configs/config.yaml
```

3. Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

## 🔍 Inference

### Single Image
```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.jpg \
    --output_dir inference/results
```

### Batch Inference
```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pth \
    --image_dir path/to/images/ \
    --output_dir inference/results
```

### SAHI Inference (for small objects)
```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.jpg \
    --sahi \
    --output_dir inference/results
```

## ⚙️ Configuration

Key configuration parameters in `configs/config.yaml`:

```yaml
model:
  architecture: "yolov8"
  backbone: "resnet50"
  fpn:
    enabled: true
  attention:
    enabled: true
    type: "cbam"

data:
  img_size: [1024, 1024]  # Larger for small objects
  annotations_format: "coco"

sahi:
  enabled: true
  slice_height: 512
  slice_width: 512

training:
  epochs: 100
  batch_size: 16
  optimizer:
    name: "AdamW"
    lr: 0.001
```

## 📈 Key Techniques for Small Object Detection

1. **Multi-Scale Feature Fusion**: FPN/PANet combines features from multiple scales
2. **Attention Mechanisms**: CBAM enhances feature representation
3. **GIoU Loss**: Better localization for small boxes
4. **Focal Loss**: Handles class imbalance
5. **SAHI**: Slicing large images for better small object detection
6. **Heavy Augmentation**: Mosaic, MixUp, geometric transforms

## 🎯 Performance Tips

- Use larger input sizes (1024x1024) for small objects
- Enable SAHI for images with very small objects
- Adjust `small_object_weight` in config to emphasize small objects
- Use lower confidence threshold for small objects
- Consider ensembling multiple models

## 📝 TODO

- [ ] Add your dataset
- [ ] Configure hyperparameters
- [ ] Train the model
- [ ] Evaluate on validation set
- [ ] Run inference on test images
- [ ] Fine-tune for your specific use case

## 🤝 Contributing

This is a showcase project demonstrating expertise in:
- Deep learning for computer vision
- PyTorch implementation
- Object detection
- Small object detection optimization
- Production-ready ML code

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the framework
- COCO dataset creators
- Research papers: FPN, PANet, YOLO, CBAM, GIoU, Focal Loss
