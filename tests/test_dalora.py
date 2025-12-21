"""
Test script for Domain-Aware LoRA (DA-LoRA) implementation.
Run with: python tests/test_dalora.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

def test_dalora_layer():
    """Test DomainAwareLoRALayer."""
    print("\n" + "="*60)
    print("Test 1: DomainAwareLoRALayer")
    print("="*60)

    from src.models.da_lora import DomainAwareLoRALayer

    batch_size = 4
    seq_len = 197  # ViT sequence length
    in_features = 384
    out_features = 384
    num_domains = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    layer = DomainAwareLoRALayer(
        in_features=in_features,
        out_features=out_features,
        num_domains=num_domains,
        rank=16
    ).to(device)

    x = torch.randn(batch_size, seq_len, in_features).to(device)
    domain_weights = F.softmax(torch.randn(batch_size, num_domains), dim=1).to(device)

    print(f"Input shape: {x.shape}")
    print(f"Domain weights: {domain_weights[0].tolist()}")

    output = layer(x, domain_weights)
    print(f"Output shape: {output.shape}")

    # Test without domain weights (uniform)
    output_uniform = layer(x, None)
    print(f"Output with uniform weights shape: {output_uniform.shape}")

    print("Test 1: PASSED")
    return True


def test_linear_with_dalora():
    """Test LinearWithDALoRA."""
    print("\n" + "="*60)
    print("Test 2: LinearWithDALoRA")
    print("="*60)

    from src.models.da_lora import LinearWithDALoRA

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_features = 384
    out_features = 384
    batch_size = 4
    num_domains = 3

    linear = torch.nn.Linear(in_features, out_features).to(device)
    dalora_linear = LinearWithDALoRA(
        linear,
        num_domains=num_domains,
        rank=16
    ).to(device)

    x = torch.randn(batch_size, in_features).to(device)
    domain_weights = F.softmax(torch.randn(batch_size, num_domains), dim=1).to(device)

    output = dalora_linear(x, domain_weights)
    print(f"Output shape: {output.shape}")

    # Check frozen weights
    assert not linear.weight.requires_grad, "Original weights should be frozen"
    print("Original weights frozen: YES")

    print("Test 2: PASSED")
    return True


def test_gradient_flow():
    """Test gradient flow through DA-LoRA."""
    print("\n" + "="*60)
    print("Test 3: Gradient Flow")
    print("="*60)

    from src.models.da_lora import LinearWithDALoRA

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_features = 384
    out_features = 384
    batch_size = 4
    num_domains = 3

    linear = torch.nn.Linear(in_features, out_features).to(device)
    dalora_linear = LinearWithDALoRA(
        linear,
        num_domains=num_domains,
        rank=16
    ).to(device)

    x = torch.randn(batch_size, in_features).to(device)
    domain_weights = F.softmax(torch.randn(batch_size, num_domains), dim=1).to(device)

    output = dalora_linear(x, domain_weights)
    loss = output.sum()
    loss.backward()

    has_grad = False
    for name, param in dalora_linear.named_parameters():
        if param.grad is not None and 'lora' in name:
            has_grad = True
            print(f"  {name}: grad_norm = {param.grad.norm():.6f}")

    if has_grad:
        print("Test 3: PASSED - Gradients flow through DA-LoRA")
        return True
    else:
        print("Test 3: FAILED - No gradients")
        return False


def test_face_recognition_model_dalora():
    """Test FaceRecognitionModel with DA-LoRA."""
    print("\n" + "="*60)
    print("Test 4: FaceRecognitionModel with DA-LoRA")
    print("="*60)

    from src.models import FaceRecognitionModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    try:
        model = FaceRecognitionModel(
            backbone_type='dinov2',
            num_classes=1000,
            use_lora=True,
            lora_rank=16,
            domain_aware=True,
            num_domains=3
        ).to(device)

        output = model(x)

        print(f"Embeddings shape: {output['embeddings'].shape}")
        print(f"Features shape: {output['features'].shape}")
        print(f"Domain logits shape: {output['domain_logits'].shape}")
        print(f"Domain weights: {output['domain_weights'][0].tolist()}")

        # Check embedding normalization
        embedding_norms = output['embeddings'].norm(dim=1)
        print(f"Embedding norms: {embedding_norms.tolist()}")

        print("Test 4: PASSED")
        return True

    except Exception as e:
        print(f"Test 4: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dalora_training_step():
    """Test a complete training step with DA-LoRA."""
    print("\n" + "="*60)
    print("Test 5: DA-LoRA Training Step")
    print("="*60)

    from src.models import FaceRecognitionModel
    from src.losses import CosFaceLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 4
    num_classes = 100
    num_domains = 3

    try:
        # Create model
        model = FaceRecognitionModel(
            backbone_type='dinov2',
            num_classes=num_classes,
            use_lora=True,
            lora_rank=16,
            domain_aware=True,
            num_domains=num_domains
        ).to(device)

        # Create loss
        loss_fn = CosFaceLoss(
            embedding_dim=512,
            num_classes=num_classes,
            scale=64.0,
            margin=0.35
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4
        )

        # Dummy input
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)

        # Forward pass
        model.train()
        output = model(x)
        embeddings = output['embeddings']
        domain_logits = output['domain_logits']

        # Compute loss
        face_loss = loss_fn(embeddings, labels)

        # Domain loss (entropy minimization for confident predictions)
        domain_probs = F.softmax(domain_logits, dim=1)
        domain_loss = -(domain_probs * torch.log(domain_probs + 1e-8)).sum(dim=1).mean()

        total_loss = face_loss + 0.1 * domain_loss

        print(f"Face loss: {face_loss.item():.4f}")
        print(f"Domain loss: {domain_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_count += 1

        print(f"Parameters with gradients: {grad_count}")

        # Optimizer step
        optimizer.step()

        print("Test 5: PASSED - Training step completed")
        return True

    except Exception as e:
        print(f"Test 5: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("DA-LoRA Implementation Tests")
    print("="*60)

    results = []

    results.append(("DomainAwareLoRALayer", test_dalora_layer()))
    results.append(("LinearWithDALoRA", test_linear_with_dalora()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("FaceRecognitionModel + DA-LoRA", test_face_recognition_model_dalora()))
    results.append(("Training Step", test_dalora_training_step()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\nAll tests PASSED! DA-LoRA is ready to use.")
        print("\nTo train with DA-LoRA, run:")
        print("  python train.py --backbone dinov2 --use-dalora --num-domains 3 --batch-size 128 --epochs 40 --wandb")
    else:
        print("\nSome tests FAILED. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
