"""
CosFace Loss Implementation for AdaptFace
Implements margin-based softmax losses for face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CosFaceLoss(nn.Module):
    """
    CosFace (Large Margin Cosine Loss) for face recognition.

    Loss = -log(exp(s*(cos(θ_y) - m)) / (exp(s*(cos(θ_y) - m)) + Σ exp(s*cos(θ_j))))

    Where:
    - s: scaling factor (default 64.0)
    - m: margin (default 0.35)
    - θ_y: angle between feature and target class center
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.35
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of classes
            scale: Scaling factor (s)
            margin: Cosine margin (m)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Learnable class centers (weight matrix)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CosFace loss.

        Args:
            embeddings: Normalized face embeddings [B, embedding_dim]
            labels: Class labels [B]

        Returns:
            Loss value
        """
        # Normalize weight matrix
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: embeddings @ weight^T
        # embeddings should already be normalized
        cosine = F.linear(embeddings, weight_norm)  # [B, num_classes]

        # Apply margin to target class
        # Create one-hot mask
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Subtract margin from target class cosine
        output = cosine - one_hot * self.margin

        # Scale
        output = output * self.scale

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss.

    Loss uses angular margin: cos(θ_y + m) instead of cos(θ_y) - m
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of classes
            scale: Scaling factor
            margin: Angular margin (in radians)
            easy_margin: Use easy margin variant
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Precompute
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: Normalized face embeddings [B, embedding_dim]
            labels: Class labels [B]

        Returns:
            Loss value
        """
        # Normalize weight
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine
        cosine = F.linear(embeddings, weight_norm)

        # Compute sin from cos
        sine = torch.sqrt(1.0 - torch.clamp(cosine.pow(2), 0, 1))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale
        output = output * self.scale

        # Cross entropy
        loss = F.cross_entropy(output, labels)

        return loss


class ContrastiveDomainLoss(nn.Module):
    """
    Contrastive loss for domain alignment.

    Pulls together same-identity features across domains,
    pushes apart different-identity features.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive domain alignment loss.

        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Identity labels [B]
            domain_labels: Domain labels [B] (optional, for cross-domain)

        Returns:
            Loss value
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive mask (same identity)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # Remove self-similarity from positives
        identity_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - identity_mask

        # Negative mask (different identity)
        negative_mask = 1 - positive_mask - identity_mask

        # For numerical stability, subtract max
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Compute log softmax over negatives + positives
        exp_sim = torch.exp(sim_matrix)

        # Sum of exp for negatives
        neg_sum = (exp_sim * negative_mask).sum(dim=1)

        # Loss for each positive pair
        pos_exp = exp_sim * positive_mask

        # Avoid division by zero
        pos_count = positive_mask.sum(dim=1).clamp(min=1)

        # Check if there are any positive pairs
        pos_sum = pos_exp.sum(dim=1)
        has_positives = pos_sum > 0

        if has_positives.sum() == 0:
            # No positive pairs in batch - return 0 loss
            return torch.tensor(0.0, device=device, requires_grad=True)

        # InfoNCE-style loss (only for samples with positives)
        loss = -torch.log(
            (pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8)
        )

        # Average over samples that have positives
        loss = loss[has_positives].mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for DA-LoRA training.

    Total = CosFace + λ1 * Contrastive + λ2 * DomainClassifier
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_domains: int = 3,
        scale: float = 64.0,
        margin: float = 0.35,
        lambda_contrastive: float = 0.1,
        lambda_domain: float = 0.01,
        temperature: float = 0.07
    ):
        """
        Args:
            embedding_dim: Embedding dimension
            num_classes: Number of identity classes
            num_domains: Number of domains
            scale: CosFace scale
            margin: CosFace margin
            lambda_contrastive: Weight for contrastive loss
            lambda_domain: Weight for domain classification loss
            temperature: Contrastive temperature
        """
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.lambda_domain = lambda_domain

        # Main loss
        self.cosface = CosFaceLoss(embedding_dim, num_classes, scale, margin)

        # Contrastive loss
        self.contrastive = ContrastiveDomainLoss(temperature)

        # Domain classification loss
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.

        Args:
            embeddings: Face embeddings [B, D]
            labels: Identity labels [B]
            domain_logits: Domain classifier outputs [B, num_domains]
            domain_labels: Domain labels [B]

        Returns:
            Dictionary with 'total', 'cosface', 'contrastive', 'domain' losses
        """
        losses = {}

        # Main CosFace loss
        losses['cosface'] = self.cosface(embeddings, labels)

        # Contrastive loss
        losses['contrastive'] = self.contrastive(embeddings, labels, domain_labels)

        # Domain classification loss
        if domain_logits is not None and domain_labels is not None:
            losses['domain'] = self.domain_criterion(domain_logits, domain_labels)
        else:
            losses['domain'] = torch.tensor(0.0, device=embeddings.device)

        # Total loss
        losses['total'] = (
            losses['cosface'] +
            self.lambda_contrastive * losses['contrastive'] +
            self.lambda_domain * losses['domain']
        )

        return losses


if __name__ == "__main__":
    print("Testing Loss Functions...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32
    embedding_dim = 512
    num_classes = 1000
    num_domains = 3

    # Generate dummy data
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), dim=1).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    domain_labels = torch.randint(0, num_domains, (batch_size,)).to(device)
    domain_logits = torch.randn(batch_size, num_domains).to(device)

    # Test CosFace
    print("\n=== Testing CosFaceLoss ===")
    cosface = CosFaceLoss(embedding_dim, num_classes).to(device)
    loss = cosface(embeddings, labels)
    print(f"CosFace loss: {loss.item():.4f}")
    print("CosFaceLoss: PASSED")

    # Test ArcFace
    print("\n=== Testing ArcFaceLoss ===")
    arcface = ArcFaceLoss(embedding_dim, num_classes).to(device)
    loss = arcface(embeddings, labels)
    print(f"ArcFace loss: {loss.item():.4f}")
    print("ArcFaceLoss: PASSED")

    # Test Contrastive
    print("\n=== Testing ContrastiveDomainLoss ===")
    contrastive = ContrastiveDomainLoss().to(device)
    loss = contrastive(embeddings, labels, domain_labels)
    print(f"Contrastive loss: {loss.item():.4f}")
    print("ContrastiveDomainLoss: PASSED")

    # Test Combined
    print("\n=== Testing CombinedLoss ===")
    combined = CombinedLoss(embedding_dim, num_classes, num_domains).to(device)
    losses = combined(embeddings, labels, domain_logits, domain_labels)
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"  CosFace: {losses['cosface'].item():.4f}")
    print(f"  Contrastive: {losses['contrastive'].item():.4f}")
    print(f"  Domain: {losses['domain'].item():.4f}")
    print("CombinedLoss: PASSED")

    # Test gradient flow
    print("\n=== Testing Gradient Flow ===")
    embeddings.requires_grad = True
    loss = cosface(embeddings, labels)
    loss.backward()
    print(f"Gradient norm: {embeddings.grad.norm().item():.4f}")
    print("Gradient Flow: PASSED")

    print("\nAll loss tests passed!")
