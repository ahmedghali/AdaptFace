"""
Training Pipeline for AdaptFace
Implements complete training loop with validation, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
import json

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    backbone: str = 'dinov2'
    num_classes: int = 10572
    embedding_dim: int = 512
    use_lora: bool = True
    lora_rank: int = 16

    # Training
    batch_size: int = 512
    num_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2

    # Loss
    cosface_scale: float = 64.0
    cosface_margin: float = 0.35

    # Data
    num_workers: int = 4
    train_data_dir: str = 'data/casia-webface/faces_webface_112x112'
    val_data_dir: str = 'data/lfw'

    # Checkpointing
    save_dir: str = 'checkpoints'
    save_every: int = 5
    keep_last: int = 3

    # Logging
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = 'AdaptFace'
    experiment_name: str = 'baseline'
    log_every: int = 100

    # Validation
    val_every: int = 1
    val_batch_size: int = 128

    # Device
    device: str = 'cuda'
    mixed_precision: bool = True


class Trainer:
    """
    Complete training pipeline for face recognition.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        """
        Args:
            model: Face recognition model
            loss_fn: Loss function (CosFace/ArcFace)
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Model and loss
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer - only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += [p for p in self.loss_fn.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.config.mixed_precision else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Setup directories and logging
        self._setup_directories()
        self._setup_logging()

        # Print training info
        self._print_training_info()

    def _setup_directories(self):
        """Create necessary directories."""
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.save_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Setup W&B and TensorBoard logging."""
        self.wandb_run = None
        self.tb_writer = None

        # W&B
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.project_name,
                    name=self.config.experiment_name,
                    config=vars(self.config),
                    reinit=True
                )
                print("W&B logging enabled")
            except Exception as e:
                print(f"W&B init failed: {e}")

        # TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"TensorBoard logging to: {self.log_dir}")

    def _print_training_info(self):
        """Print training configuration."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        if self.val_loader:
            print(f"Val samples: {len(self.val_loader.dataset):,}")
        print("="*60 + "\n")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        self.loss_fn.train()

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    output = self.model(images)
                    embeddings = output['embeddings'] if isinstance(output, dict) else output
                    loss = self.loss_fn(embeddings, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                embeddings = output['embeddings'] if isinstance(output, dict) else output
                loss = self.loss_fn(embeddings, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} LR: {lr:.6f}")

                # W&B logging
                if self.wandb_run:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': lr,
                        'train/step': self.global_step
                    })

                # TensorBoard logging
                if self.tb_writer:
                    self.tb_writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.tb_writer.add_scalar('train/lr', lr, self.global_step)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches

        print(f"Epoch {self.current_epoch} completed in {epoch_time:.1f}s - Avg Loss: {avg_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation on LFW pairs.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}

        from src.evaluation import evaluate_lfw_pairs

        self.model.eval()

        results = evaluate_lfw_pairs(
            self.model,
            self.val_loader,
            self.device
        )

        # Log results
        print(f"\nValidation Results:")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  K-fold: {results['kfold_accuracy_mean']*100:.2f}% (+/- {results['kfold_accuracy_std']*100:.2f}%)")
        print(f"  AUC: {results['auc']:.4f}")

        # W&B logging
        if self.wandb_run:
            wandb.log({
                'val/accuracy': results['accuracy'],
                'val/kfold_accuracy': results['kfold_accuracy_mean'],
                'val/auc': results['auc'],
                'val/epoch': self.current_epoch
            })

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('val/accuracy', results['accuracy'], self.current_epoch)
            self.tb_writer.add_scalar('val/auc', results['auc'], self.current_epoch)

        return results

    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': vars(self.config),
            'history': self.history
        }

        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.save_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: int(x.stem.split('_')[-1])
        )

        while len(checkpoints) > self.config.keep_last:
            checkpoints[0].unlink()
            checkpoints.pop(0)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = checkpoint.get('history', self.history)

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train(self):
        """
        Run full training loop.
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        start_epoch = self.current_epoch
        total_start = time.time()

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*40}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*40}")

            # Train
            avg_loss = self.train_epoch()
            self.history['train_loss'].append(avg_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Update scheduler
            self.scheduler.step()

            # Validation
            if self.val_loader and (epoch + 1) % self.config.val_every == 0:
                val_results = self.validate()
                accuracy = val_results.get('accuracy', 0)
                self.history['val_accuracy'].append(accuracy)

                # Check for best model
                is_best = accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = accuracy
                    print(f"New best accuracy: {accuracy*100:.2f}%")
            else:
                is_best = False

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

        total_time = time.time() - total_start
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best accuracy: {self.best_accuracy*100:.2f}%")
        print("="*60)

        # Save final checkpoint
        self.save_checkpoint(filename='final_model.pt')

        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Close logging
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            wandb.finish()


def create_trainer(
    config: TrainingConfig = None,
    resume_from: str = None
) -> Trainer:
    """
    Factory function to create trainer with all components.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Configured Trainer instance
    """
    from src.models import FaceRecognitionModel
    from src.losses import CosFaceLoss
    from src.data import CASIAWebFaceFromList, ValidationPairDataset
    from src.data import get_train_transforms, get_val_transforms

    config = config or TrainingConfig()

    print("Creating training components...")

    # Create model
    model = FaceRecognitionModel(
        backbone_type=config.backbone,
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank
    )

    # Create loss
    loss_fn = CosFaceLoss(
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
        scale=config.cosface_scale,
        margin=config.cosface_margin
    )

    # Create train dataloader
    train_dataset = CASIAWebFaceFromList(
        config.train_data_dir,
        transform=get_train_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Create val dataloader
    val_dataset = ValidationPairDataset(
        config.val_data_dir,
        transform=get_val_transforms()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Resume if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)

    return trainer


if __name__ == "__main__":
    print("Testing Trainer Module...")

    # Test with dummy data
    from torch.utils.data import TensorDataset

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3*224*224, 512)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return {'embeddings': torch.nn.functional.normalize(self.fc(x), dim=1)}

    # Create dummy loss
    class DummyLoss(nn.Module):
        def forward(self, embeddings, labels):
            return torch.tensor(1.0, requires_grad=True)

    # Create dummy data
    n_samples = 100
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, 10, (n_samples,))
    train_dataset = TensorDataset(images, labels)
    train_loader = DataLoader(train_dataset, batch_size=16)

    # Create dummy validation data
    n_val = 50
    img1 = torch.randn(n_val, 3, 224, 224)
    img2 = torch.randn(n_val, 3, 224, 224)
    val_labels = torch.randint(0, 2, (n_val,))
    val_dataset = TensorDataset(img1, img2, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Create config
    config = TrainingConfig(
        num_epochs=2,
        batch_size=16,
        use_wandb=False,
        use_tensorboard=False,
        save_dir='test_checkpoints',
        log_every=10
    )

    # Create trainer
    model = DummyModel()
    loss_fn = DummyLoss()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Run training
    print("\nRunning test training...")
    trainer.train()

    # Cleanup
    import shutil
    if Path('test_checkpoints').exists():
        shutil.rmtree('test_checkpoints')

    print("\nTrainer Module: PASSED")