"""
Training script for small object detection
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from models.detector import build_detector
from dataloader import build_dataloader
from utils.losses import DetectionLoss, YOLOLoss
from utils.metrics import MeanAveragePrecision


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

        # TensorBoard
        if log_config['use_tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
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

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)

            # Compute loss
            losses = self.criterion(predictions, targets)
            loss = losses['total_loss']

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })

            # Log to tensorboard
            if self.writer and batch_idx % 10 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/cls_loss', losses['cls_loss'].item(), global_step)
                self.writer.add_scalar('train/bbox_loss', losses['bbox_loss'].item(), global_step)
                self.writer.add_scalar('train/obj_loss', losses['obj_loss'].item(), global_step)

        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
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

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, is_best=False):
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
            print(f"Saved best checkpoint to {best_path}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if epoch % self.config['validation']['interval'] == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

                # Log to tensorboard
                if self.writer:
                    self.writer.add_scalar('val/loss', val_loss, epoch)

                # Save checkpoint
                is_best = False  # TODO: Implement mAP computation
                self.save_checkpoint(is_best=is_best)

        print("\nTraining completed!")

        if self.writer:
            self.writer.close()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train small object detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
