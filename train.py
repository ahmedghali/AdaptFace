#!/usr/bin/env python3
"""
AdaptFace Training Script
Train face recognition model with LoRA on CASIA-WebFace.
"""

import argparse
import torch
import platform
from pathlib import Path
from multiprocessing import freeze_support

from src.models import FaceRecognitionModel
from src.losses import CosFaceLoss
from src.data import CASIAWebFaceFromList, ValidationPairDataset
from src.data import get_train_transforms, get_val_transforms
from src.training import Trainer, TrainingConfig
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train AdaptFace model')

    # Model
    parser.add_argument('--backbone', type=str, default='dinov2',
                        choices=['dinov2', 'clip'], help='Backbone model')
    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=None,
                        help='LoRA alpha scaling (default: 2x rank)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Learning rate warmup epochs')
    parser.add_argument('--no-lora', action='store_true',
                        help='Disable LoRA (full fine-tuning)')

    # DA-LoRA (Domain-Aware LoRA)
    parser.add_argument('--use-dalora', action='store_true',
                        help='Enable Domain-Aware LoRA (DA-LoRA)')
    parser.add_argument('--num-domains', type=int, default=3,
                        help='Number of domains for DA-LoRA (default: 3)')
    parser.add_argument('--domain-loss-weight', type=float, default=0.1,
                        help='Weight for domain classification loss')

    # Training
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')

    # Loss
    parser.add_argument('--margin', type=float, default=0.35,
                        help='CosFace margin')
    parser.add_argument('--scale', type=float, default=64.0,
                        help='CosFace scale')

    # Data
    parser.add_argument('--train-data', type=str,
                        default='data/casia-webface/faces_webface_112x112',
                        help='Training data directory')
    parser.add_argument('--val-data', type=str, default='data/lfw',
                        help='Validation data directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for debugging)')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--project', type=str, default='AdaptFace',
                        help='W&B project name')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set lora_alpha default (2x rank)
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank * 2.0

    # Set experiment name
    if args.name is None:
        if args.use_dalora:
            args.name = f"{args.backbone}_dalora{args.lora_rank}_d{args.num_domains}_bs{args.batch_size}"
        else:
            args.name = f"{args.backbone}_lora{args.lora_rank}_bs{args.batch_size}"

    print("\n" + "="*60)
    print("AdaptFace Training")
    if args.use_dalora:
        print(f"Mode: DA-LoRA ({args.num_domains} domains)")
    print("="*60)

    # Create config
    config = TrainingConfig(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        domain_aware=args.use_dalora,
        num_domains=args.num_domains,
        domain_loss_weight=args.domain_loss_weight,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cosface_margin=args.margin,
        cosface_scale=args.scale,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        use_tensorboard=True,
        project_name=args.project,
        experiment_name=args.name,
        device=args.device,
        mixed_precision=not args.no_amp
    )

    # Check data directories
    train_path = Path(config.train_data_dir)
    val_path = Path(config.val_data_dir)

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        return

    if not val_path.exists():
        print(f"WARNING: Validation data not found: {val_path}")
        val_loader = None
    else:
        # Create validation dataloader
        print(f"\nLoading validation data from: {val_path}")
        val_dataset = ValidationPairDataset(
            data_dir=str(val_path),
            dataset_name='lfw',
            transform=get_val_transforms()
        )
        # Use num_workers=0 for validation on Windows to avoid multiprocessing issues
        val_num_workers = 0 if platform.system() == 'Windows' else config.num_workers
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True
        )

    # Create training dataloader
    print(f"\nLoading training data from: {train_path}")
    train_dataset = CASIAWebFaceFromList(
        data_dir=str(train_path),
        transform=get_train_transforms(),
        max_samples=args.max_samples
    )

    config.num_classes = train_dataset.num_classes
    print(f"Training dataset: {len(train_dataset):,} samples, {config.num_classes} classes")

    # Reduce workers on Windows to avoid multiprocessing issues
    train_num_workers = min(config.num_workers, 2) if platform.system() == 'Windows' else config.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Create model
    if config.domain_aware:
        print(f"\nCreating model: {config.backbone} with DA-LoRA (rank={config.lora_rank}, alpha={config.lora_alpha}, domains={config.num_domains})")
    else:
        print(f"\nCreating model: {config.backbone} with LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})")
    print(f"Warmup: {config.warmup_epochs} epochs")

    model = FaceRecognitionModel(
        backbone_type=config.backbone,
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        domain_aware=config.domain_aware,
        num_domains=config.num_domains
    )

    # Create loss
    loss_fn = CosFaceLoss(
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
        scale=config.cosface_scale,
        margin=config.cosface_margin
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
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()