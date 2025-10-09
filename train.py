"""
Training script for small object detection with WandB integration
"""
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
from pathlib import Path

from models.detector import build_detector
from dataloader import build_dataloader
from utils.losses import DetectionLoss, YOLOLoss
from utils.metrics import MeanAveragePrecision
from wandb_logger import WandbLogger


class Trainer:
    """Trainer class for small object detection"""

    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(
            self.config['training']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Build model
        print("Building model...")
        self.model = build_detector(self.config)
        self.model = self.model.to(self.device)

        # Multi-GPU support
        if self.config['training']['multi_gpu'] and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Build dataloaders
        print("Building dataloaders...")
        self.train_loader = build_dataloader(self.config, split='train')
        self.val_loader = build_dataloader(self.config, split='val')

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Build loss function
        self.criterion = DetectionLoss(
            num_classes=self.config['data']['num_classes'],
            bbox_loss_weight=self.config['training']['loss']['bbox_loss_weight'],
            cls_loss_weight=self.config['training']['loss']['cls_loss_weight'],
            small_object_weight=self.config['training']['loss']['small_object_weight']
        )

        # Mixed precision training (AMP)
        self.use_amp = self.config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Mixed Precision Training (AMP) enabled - saves ~40% GPU memory")

        # Gradient accumulation
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            print(f"Gradient Accumulation: {self.gradient_accumulation_steps} steps")
            print(f"Effective batch size: {self.config['training']['batch_size'] * self.gradient_accumulation_steps}")

        # Memory optimization
        self.empty_cache_steps = self.config['training'].get('empty_cache_every_n_steps', 0)

        # Setup logging
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_map = 0.0

    def _build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']['optimizer']
        name = opt_config['name']

        if name == 'Adam':
            return Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif name == 'AdamW':
            return AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif name == 'SGD':
            return SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config['training']['scheduler']
        name = sched_config['name']

        if name == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config['T_0'],
                T_mult=sched_config['T_mult'],
                eta_min=sched_config['eta_min']
            )
        elif name == 'OneCycleLR':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['optimizer']['lr'],
                epochs=self.config['training']['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    def setup_logging(self):
        """Setup logging and checkpointing"""
        log_config = self.config['logging']
        self.log_dir = Path(log_config['log_dir']) / log_config['experiment_name']
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # WandB Logger
        if log_config['use_wandb']:
            print("Initializing WandB logger...")
            self.wandb_logger = WandbLogger(self.config, model=self.model)
            self.writer = None
        # TensorBoard (fallback)
        elif log_config.get('use_tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            self.wandb_logger = None
        else:
            self.writer = None
            self.wandb_logger = None

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch with AMP and gradient accumulation support"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            targets = {
                'bboxes': batch['bboxes'],
                'labels': batch['labels']
            }

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                loss = losses['total_loss']

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics (scale back for display)
            total_loss += loss.item() * self.gradient_accumulation_steps

            # Clear CUDA cache periodically
            if self.empty_cache_steps > 0 and (batch_idx + 1) % self.empty_cache_steps == 0:
                torch.cuda.empty_cache()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })

            # Log to WandB or TensorBoard
            if batch_idx % 10 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx

                metrics = {
                    'loss': loss.item(),
                    'cls_loss': losses['cls_loss'].item(),
                    'bbox_loss': losses['bbox_loss'].item(),
                    'obj_loss': losses['obj_loss'].item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }

                if self.wandb_logger:
                    self.wandb_logger.log_metrics(metrics, step=global_step, prefix='train/')
                elif self.writer:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, global_step)

        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, log_predictions=False):
        """
        Validate the model

        Args:
            log_predictions: Whether to log prediction visualizations to WandB
        """
        self.model.eval()
        total_loss = 0

        # For mAP computation
        metric = MeanAveragePrecision(
            num_classes=self.config['data']['num_classes'],
            iou_threshold=self.config['validation']['iou_threshold']
        )

        # For prediction visualization
        vis_images = []
        vis_gt_boxes = []
        vis_gt_labels = []
        vis_pred_boxes = []
        vis_pred_labels = []
        vis_pred_scores = []

        max_vis_images = 16 if log_predictions else 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            targets = {
                'bboxes': batch['bboxes'],
                'labels': batch['labels']
            }

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            losses = self.criterion(predictions, targets)
            loss = losses['total_loss']

            total_loss += loss.item()

            # Get predictions for mAP computation
            # predictions is expected to have 'boxes', 'scores', 'labels'
            pred_boxes = predictions.get('boxes', [])
            pred_scores = predictions.get('scores', [])
            pred_labels = predictions.get('labels', [])

            # Update mAP metric
            for i in range(len(images)):
                gt_boxes_i = targets['bboxes'][i].cpu().numpy() if isinstance(targets['bboxes'][i], torch.Tensor) else targets['bboxes'][i]
                gt_labels_i = targets['labels'][i].cpu().numpy() if isinstance(targets['labels'][i], torch.Tensor) else targets['labels'][i]

                pred_boxes_i = pred_boxes[i].cpu().numpy() if len(pred_boxes) > 0 else np.array([])
                pred_scores_i = pred_scores[i].cpu().numpy() if len(pred_scores) > 0 else np.array([])
                pred_labels_i = pred_labels[i].cpu().numpy() if len(pred_labels) > 0 else np.array([])

                metric.update(
                    pred_boxes_i, pred_scores_i, pred_labels_i,
                    gt_boxes_i, gt_labels_i
                )

                # Collect for visualization
                if log_predictions and len(vis_images) < max_vis_images:
                    vis_images.append(images[i].cpu())
                    vis_gt_boxes.append(gt_boxes_i)
                    vis_gt_labels.append(gt_labels_i)
                    vis_pred_boxes.append(pred_boxes_i)
                    vis_pred_labels.append(pred_labels_i)
                    vis_pred_scores.append(pred_scores_i)

            pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        map_score = metric.compute()

        # Log predictions to WandB
        if log_predictions and self.wandb_logger and len(vis_images) > 0:
            print("Logging predictions to WandB...")

            # Log images with boxes
            self.wandb_logger.log_images_with_boxes(
                images=vis_images,
                gt_boxes=vis_gt_boxes,
                gt_labels=vis_gt_labels,
                pred_boxes=vis_pred_boxes,
                pred_labels=vis_pred_labels,
                pred_scores=vis_pred_scores,
                step=self.current_epoch,
                split='val',
                max_images=max_vis_images
            )

            # Log prediction grid
            self.wandb_logger.log_prediction_grid(
                images=vis_images,
                pred_boxes=vis_pred_boxes,
                pred_labels=vis_pred_labels,
                pred_scores=vis_pred_scores,
                gt_boxes=vis_gt_boxes,
                gt_labels=vis_gt_labels,
                step=self.current_epoch,
                split='val',
                num_images=min(16, len(vis_images))
            )

        return avg_loss, map_score

    def save_checkpoint(self, is_best=False, map_score=0.0):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_map': self.best_map,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path} (mAP: {map_score:.4f})")

            # Log to WandB
            if self.wandb_logger:
                self.wandb_logger.log_model_checkpoint(
                    str(best_path),
                    metadata={
                        'epoch': self.current_epoch,
                        'map': map_score,
                        'best_map': self.best_map
                    }
                )

    def train(self):
        """Main training loop with WandB integration"""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        # Log initial info
        if self.wandb_logger:
            print(f"\nWandB Run: {self.wandb_logger.run.url}")
            print("Track your training in real-time at the URL above!\n")

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            print('='*60)

            # Train
            train_loss = self.train_epoch()
            print(f"\nTrain Loss: {train_loss:.4f}")

            # Log training metrics
            train_metrics = {
                'loss': train_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

            if self.wandb_logger:
                self.wandb_logger.log_metrics(train_metrics, step=epoch, prefix='epoch/train/')
            elif self.writer:
                self.writer.add_scalar('epoch/train/loss', train_loss, epoch)

            # Validate
            if epoch % self.config['validation']['interval'] == 0:
                # Log predictions every N epochs
                log_predictions = (epoch % 5 == 0) and self.wandb_logger is not None

                val_loss, map_score = self.validate(log_predictions=log_predictions)

                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val mAP: {map_score:.4f}")

                # Log validation metrics
                val_metrics = {
                    'loss': val_loss,
                    'mAP': map_score,
                    'mAP_50': map_score  # Assuming this is mAP@0.5
                }

                if self.wandb_logger:
                    self.wandb_logger.log_metrics(val_metrics, step=epoch, prefix='epoch/val/')
                elif self.writer:
                    self.writer.add_scalar('epoch/val/loss', val_loss, epoch)
                    self.writer.add_scalar('epoch/val/mAP', map_score, epoch)

                # Check if best model
                is_best = map_score > self.best_map
                if is_best:
                    self.best_map = map_score
                    print(f"🎉 New best mAP: {self.best_map:.4f}")

                # Save checkpoint
                self.save_checkpoint(is_best=is_best, map_score=map_score)

        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        print(f"Best mAP: {self.best_map:.4f}")

        if self.wandb_logger:
            print(f"\nFinal WandB Run: {self.wandb_logger.run.url}")
            print("Check WandB for full training history and visualizations!")
            self.wandb_logger.finish()
        elif self.writer:
            self.writer.close()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train small object detector with WandB')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override WandB setting if specified
    if args.no_wandb:
        config['logging']['use_wandb'] = False
        print("WandB logging disabled via command line")

    # Save config back to temporary variable
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    # Create trainer and start training
    trainer = Trainer(temp_config_path)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint.get('epoch', 0) + 1
        trainer.best_map = checkpoint.get('best_map', 0.0)
        print(f"Resumed from epoch {trainer.current_epoch}, best mAP: {trainer.best_map:.4f}")

    trainer.train()

    # Clean up temp config
    import os
    os.unlink(temp_config_path)


if __name__ == "__main__":
    main()
