"""
LFW Evaluation Module for AdaptFace
Implements face verification evaluation on LFW and other pair datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from scipy import interpolate


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embeddings [B, D]
        emb2: Second embeddings [B, D]

    Returns:
        Similarity scores [B]
    """
    # Normalize (should already be normalized, but ensure)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    # Cosine similarity
    return (emb1 * emb2).sum(dim=1)


def compute_accuracy(
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> float:
    """Compute accuracy at a given threshold."""
    predictions = (similarities > threshold).astype(int)
    return (predictions == labels).mean()


def find_best_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Find the best threshold for verification.

    Returns:
        Tuple of (best_threshold, best_accuracy)
    """
    thresholds = np.linspace(-1, 1, num_thresholds)

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy = compute_accuracy(similarities, labels, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def compute_roc_auc(
    similarities: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ROC curve and AUC.

    Returns:
        Tuple of (auc_score, fpr_array, tpr_array)
    """
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def compute_tar_at_far(
    similarities: np.ndarray,
    labels: np.ndarray,
    far_targets: List[float] = [1e-3, 1e-2, 1e-1]
) -> Dict[str, float]:
    """
    Compute TAR (True Accept Rate) at specified FAR (False Accept Rate).

    Returns:
        Dictionary with TAR@FAR values
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)

    results = {}
    for far in far_targets:
        # Interpolate to find TAR at this FAR
        if far < fpr.min() or far > fpr.max():
            tar = 0.0
        else:
            f = interpolate.interp1d(fpr, tpr)
            tar = float(f(far))
        results[f'TAR@FAR={far}'] = tar

    return results


class LFWEvaluator:
    """
    Evaluator for LFW-style pair verification datasets.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device = None,
        num_folds: int = 10
    ):
        """
        Args:
            model: Face recognition model
            dataloader: DataLoader for validation pairs
            device: Compute device
            num_folds: Number of folds for cross-validation
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_folds = num_folds

    @torch.no_grad()
    def extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings for all pairs.

        Returns:
            Tuple of (emb1_array, emb2_array, labels_array)
        """
        self.model.eval()

        all_emb1 = []
        all_emb2 = []
        all_labels = []

        for batch in self.dataloader:
            img1, img2, labels = batch
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            # Get embeddings
            if hasattr(self.model, 'get_embeddings'):
                emb1 = self.model.get_embeddings(img1)
                emb2 = self.model.get_embeddings(img2)
            else:
                output1 = self.model(img1)
                output2 = self.model(img2)
                emb1 = output1['embeddings'] if isinstance(output1, dict) else output1
                emb2 = output2['embeddings'] if isinstance(output2, dict) else output2

            all_emb1.append(emb1.cpu().numpy())
            all_emb2.append(emb2.cpu().numpy())
            all_labels.append(labels.numpy())

        return (
            np.concatenate(all_emb1, axis=0),
            np.concatenate(all_emb2, axis=0),
            np.concatenate(all_labels, axis=0)
        )

    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation.

        Returns:
            Dictionary with evaluation metrics
        """
        # Extract embeddings
        emb1, emb2, labels = self.extract_embeddings()

        # Compute similarities
        similarities = np.sum(emb1 * emb2, axis=1)

        # Find best threshold and accuracy
        best_threshold, best_accuracy = find_best_threshold(similarities, labels)

        # Compute AUC
        roc_auc, _, _ = compute_roc_auc(similarities, labels)

        # Compute TAR@FAR
        tar_far = compute_tar_at_far(similarities, labels)

        # K-fold cross-validation for more robust accuracy
        fold_accuracies = self._kfold_accuracy(similarities, labels)
        kfold_acc_mean = np.mean(fold_accuracies)
        kfold_acc_std = np.std(fold_accuracies)

        results = {
            'accuracy': best_accuracy,
            'threshold': best_threshold,
            'auc': roc_auc,
            'kfold_accuracy_mean': kfold_acc_mean,
            'kfold_accuracy_std': kfold_acc_std,
            **tar_far
        }

        return results

    def _kfold_accuracy(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> List[float]:
        """K-fold cross-validation for accuracy."""
        n = len(labels)
        fold_size = n // self.num_folds
        accuracies = []

        indices = np.arange(n)

        for fold in range(self.num_folds):
            # Split into train/test for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.num_folds - 1 else n

            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_start:test_end] = True
            train_mask = ~test_mask

            # Find best threshold on train
            train_sims = similarities[train_mask]
            train_labels = labels[train_mask]
            threshold, _ = find_best_threshold(train_sims, train_labels)

            # Evaluate on test
            test_sims = similarities[test_mask]
            test_labels = labels[test_mask]
            accuracy = compute_accuracy(test_sims, test_labels, threshold)
            accuracies.append(accuracy)

        return accuracies


def evaluate_lfw_pairs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Convenience function for LFW evaluation.

    Args:
        model: Face recognition model
        dataloader: DataLoader for LFW pairs
        device: Compute device

    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = LFWEvaluator(model, dataloader, device)
    return evaluator.evaluate()


if __name__ == "__main__":
    print("Testing LFW Evaluation Module...")

    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 512)

        def get_embeddings(self, x):
            return F.normalize(torch.randn(x.shape[0], 512), dim=1)

    from torch.utils.data import TensorDataset

    # Create dummy pairs
    n_pairs = 100
    img1 = torch.randn(n_pairs, 3, 224, 224)
    img2 = torch.randn(n_pairs, 3, 224, 224)
    labels = torch.randint(0, 2, (n_pairs,))

    dataset = TensorDataset(img1, img2, labels)
    dataloader = DataLoader(dataset, batch_size=32)

    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    results = evaluate_lfw_pairs(model, dataloader, device)

    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    print("\nLFW Evaluation Module: PASSED")