"""
Evaluation script for AdaptFace models.
Evaluates trained models on face verification benchmarks.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --benchmark cfp-fp
    python evaluate.py --checkpoint checkpoints/best_model.pt --benchmark agedb-30
    python evaluate.py --checkpoint checkpoints/best_model.pt --benchmark all
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json

from torch.utils.data import DataLoader

from src.models import FaceRecognitionModel
from src.data import ValidationPairDataset, get_val_transforms
from src.evaluation.lfw_eval import LFWEvaluator


# Available benchmarks
BENCHMARKS = {
    'lfw': {
        'path': 'data/lfw',
        'name': 'lfw',
        'description': 'Labeled Faces in the Wild - General verification'
    },
    'cfp-fp': {
        'path': 'data/cfp-fp',
        'name': 'cfp_fp',
        'description': 'Celebrities Frontal-Profile - Pose variation'
    },
    'agedb-30': {
        'path': 'data/agedb-30',
        'name': 'agedb_30',
        'description': 'Age Database - Cross-age verification'
    },
    'calfw': {
        'path': 'data/calfw',
        'name': 'calfw',
        'description': 'Cross-Age LFW - Age variation'
    },
    'cplfw': {
        'path': 'data/cplfw',
        'name': 'cplfw',
        'description': 'Cross-Pose LFW - Pose variation'
    }
}


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    backbone_name = config.get('backbone', 'dinov2')
    lora_rank = config.get('lora_rank', 16)

    print(f"  Backbone: {backbone_name}")
    print(f"  LoRA rank: {lora_rank}")

    # Create full model with same config as training
    model = FaceRecognitionModel(
        backbone_type=backbone_name,
        embedding_dim=512,
        num_classes=10572,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=float(lora_rank),
        lora_target_modules=['qkv', 'proj']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully!")

    return model


def evaluate_benchmark(
    model: nn.Module,
    benchmark_key: str,
    device: torch.device,
    batch_size: int = 64
) -> dict:
    """Evaluate model on a specific benchmark."""

    benchmark = BENCHMARKS[benchmark_key]
    print(f"\n{'='*60}")
    print(f"Evaluating on: {benchmark_key.upper()}")
    print(f"Description: {benchmark['description']}")
    print(f"{'='*60}")

    # Check if data exists
    data_path = Path(benchmark['path'])
    if not data_path.exists():
        print(f"  ERROR: Data not found at {data_path}")
        return None

    # Load dataset
    dataset = ValidationPairDataset(
        data_dir=str(data_path),
        dataset_name=benchmark['name'],
        transform=get_val_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    # Evaluate
    evaluator = LFWEvaluator(model, dataloader, device)
    results = evaluator.evaluate()

    # Print results
    print(f"\nResults for {benchmark_key.upper()}:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  K-fold: {results['kfold_accuracy_mean']*100:.2f}% (+/- {results['kfold_accuracy_std']*100:.2f}%)")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")

    if 'TAR@FAR=0.001' in results:
        print(f"  TAR@FAR=0.1%: {results['TAR@FAR=0.001']*100:.2f}%")
    if 'TAR@FAR=0.01' in results:
        print(f"  TAR@FAR=1%: {results['TAR@FAR=0.01']*100:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate AdaptFace model on benchmarks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--benchmark', type=str, default='cfp-fp',
                        choices=list(BENCHMARKS.keys()) + ['all'],
                        help='Benchmark to evaluate on')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Determine which benchmarks to run
    if args.benchmark == 'all':
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        benchmarks_to_run = [args.benchmark]

    # Run evaluations
    all_results = {}
    for benchmark_key in benchmarks_to_run:
        results = evaluate_benchmark(model, benchmark_key, device, args.batch_size)
        if results:
            all_results[benchmark_key] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n| Benchmark | Accuracy | AUC |")
    print(f"|-----------|----------|-----|")
    for bench, res in all_results.items():
        print(f"| {bench:9} | {res['accuracy']*100:6.2f}% | {res['auc']:.4f} |")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"evaluation_results_{timestamp}.json")

    # Prepare JSON-serializable results
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'device': str(device),
        'benchmarks': {}
    }

    for bench, res in all_results.items():
        json_results['benchmarks'][bench] = {
            'accuracy': float(res['accuracy']),
            'accuracy_percent': float(res['accuracy'] * 100),
            'kfold_mean': float(res['kfold_accuracy_mean']),
            'kfold_std': float(res['kfold_accuracy_std']),
            'auc': float(res['auc']),
            'threshold': float(res['threshold'])
        }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
