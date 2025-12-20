"""
Environment Test Script for AdaptFace Project
Run this to verify all dependencies are correctly installed.
"""

import sys

def test_pytorch():
    print("=" * 50)
    print("Testing PyTorch and CUDA...")
    print("=" * 50)
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"GPU computation test: PASSED (matrix multiplication)")
    return True

def test_transformers():
    print("\n" + "=" * 50)
    print("Testing Transformers and timm...")
    print("=" * 50)
    import transformers
    import timm
    print(f"Transformers version: {transformers.__version__}")
    print(f"timm version: {timm.__version__}")

    # List available DINOv2 models
    dino_models = [m for m in timm.list_models() if 'dino' in m.lower()]
    print(f"Available DINOv2 models in timm: {len(dino_models)}")
    return True

def test_peft():
    print("\n" + "=" * 50)
    print("Testing PEFT (LoRA library)...")
    print("=" * 50)
    import peft
    print(f"PEFT version: {peft.__version__}")
    print("LoRA support: Available")
    return True

def test_visualization():
    print("\n" + "=" * 50)
    print("Testing Visualization libraries...")
    print("=" * 50)
    import matplotlib
    import seaborn
    import cv2
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {seaborn.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    return True

def test_ml_tools():
    print("\n" + "=" * 50)
    print("Testing ML tools...")
    print("=" * 50)
    import numpy as np
    import scipy
    import sklearn
    import pandas as pd
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {scipy.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Pandas version: {pd.__version__}")
    return True

def test_experiment_tracking():
    print("\n" + "=" * 50)
    print("Testing Experiment tracking...")
    print("=" * 50)
    import wandb
    import tensorboard
    print(f"W&B version: {wandb.__version__}")
    print("TensorBoard: Available")
    return True

def main():
    print("\n" + "#" * 60)
    print("#  AdaptFace Environment Test")
    print("#" * 60)

    tests = [
        ("PyTorch & CUDA", test_pytorch),
        ("Transformers & timm", test_transformers),
        ("PEFT (LoRA)", test_peft),
        ("Visualization", test_visualization),
        ("ML Tools", test_ml_tools),
        ("Experiment Tracking", test_experiment_tracking),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASSED"))
        except Exception as e:
            results.append((name, f"FAILED: {e}"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, status in results:
        icon = "[OK]" if "PASSED" in status else "[X]"
        print(f"{icon} {name}: {status}")
        if "FAILED" in status:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! Environment is ready.")
    else:
        print("Some tests FAILED. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())