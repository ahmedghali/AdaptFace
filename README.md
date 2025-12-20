# AdaptFace

**Few-Shot Adaptation of Foundation Models for Cross-Domain Face Recognition via Domain-Aware LoRA**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AdaptFace is a research project that implements **Domain-Aware Low-Rank Adaptation (DA-LoRA)** for adapting foundation models (DINOv2, CLIP) to face recognition tasks with minimal training data. The key innovation is using domain-specific LoRA modules weighted by learned domain probabilities to improve generalization across different imaging conditions (indoor, outdoor, thermal, etc.).

<p align="center">
  <img src="docs/architecture.png" alt="AdaptFace Architecture" width="700">
</p>

## Key Features

- **Foundation Model Backbones**: Support for DINOv2 ViT-S/14 and CLIP ViT-B/16
- **Parameter-Efficient Fine-Tuning**: Only ~2-4% of parameters are trained using LoRA
- **Domain-Aware LoRA**: Multiple domain-specific LoRA modules with learned weighting
- **Margin-Based Losses**: CosFace and ArcFace loss implementations
- **Contrastive Domain Alignment**: Cross-domain feature alignment loss
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Experiment Tracking**: Weights & Biases and TensorBoard integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image (224×224)                     │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              Foundation Model (DINOv2 / CLIP) [Frozen]           │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  Transformer Block + LoRA                                │  │
│    │  ┌─────────┐   ┌─────────┐   ┌─────────┐                │  │
│    │  │ LoRA_1  │   │ LoRA_2  │   │ LoRA_K  │  Domain-Aware  │  │
│    │  │(Indoor) │   │(Outdoor)│   │(Thermal)│  LoRA Modules  │  │
│    │  └────┬────┘   └────┬────┘   └────┬────┘                │  │
│    │       │             │             │                      │  │
│    │       └──────────┬──┴─────────────┘                      │  │
│    │                  │ Weighted Sum (domain probs)           │  │
│    │                  ▼                                       │  │
│    │           ┌─────────────┐                                │  │
│    │           │  Combined   │                                │  │
│    │           │   Output    │                                │  │
│    │           └─────────────┘                                │  │
│    └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Face Embedding (384/768-d)                   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CosFace / ArcFace Loss                        │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AdaptFace.git
cd AdaptFace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate  # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

## Dataset Preparation

### Training Data

Download CASIA-WebFace dataset and organize as follows:

```
data/
├── casia-webface/
│   └── faces_webface_112x112/
│       ├── train.rec
│       ├── train.idx
│       └── property
```

### Validation Benchmarks

Download and organize validation datasets:

```
data/
├── lfw/
│   ├── pairs.txt
│   └── images/
├── agedb_30/
│   ├── pairs.txt
│   └── images/
├── cfp_fp/
│   ├── pairs.txt
│   └── images/
└── ...
```

See [datasets_guide.md](datasets_guide.md) for detailed download instructions.

## Usage

### Training

**Basic training with DINOv2:**

```bash
python train.py --backbone dinov2 --batch-size 256 --epochs 40
```

**Training with Weights & Biases logging:**

```bash
python train.py \
    --backbone dinov2 \
    --lora-rank 16 \
    --batch-size 256 \
    --epochs 40 \
    --lr 1e-4 \
    --wandb \
    --project AdaptFace \
    --name dinov2_baseline
```

**Training with CLIP backbone:**

```bash
python train.py --backbone clip --batch-size 128 --epochs 40
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `dinov2` | Backbone model (`dinov2` or `clip`) |
| `--lora-rank` | `16` | LoRA rank |
| `--batch-size` | `256` | Training batch size |
| `--epochs` | `40` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `0.05` | AdamW weight decay |
| `--margin` | `0.35` | CosFace margin |
| `--scale` | `64.0` | CosFace scale |
| `--wandb` | `False` | Enable W&B logging |
| `--resume` | `None` | Resume from checkpoint |

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --benchmark lfw
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir runs/

# Weights & Biases
# Visit https://wandb.ai/your-project
```

## Project Structure

```
AdaptFace/
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
├── test_environment.py         # Environment verification
│
├── src/
│   ├── models/
│   │   ├── backbone.py         # DINOv2 & CLIP backbones
│   │   ├── lora.py             # Standard & Domain-Aware LoRA
│   │   └── face_model.py       # Complete FR model
│   │
│   ├── losses/
│   │   └── cosface.py          # CosFace, ArcFace, Contrastive losses
│   │
│   ├── data/
│   │   ├── dataset.py          # Dataset classes
│   │   ├── preprocessing.py    # Face detection & alignment
│   │   └── transforms.py       # Data augmentation
│   │
│   ├── training/
│   │   └── trainer.py          # Training loop
│   │
│   └── evaluation/
│       └── lfw_eval.py         # LFW benchmark evaluation
│
├── data/                       # Datasets (not tracked)
├── checkpoints/                # Model checkpoints (not tracked)
└── runs/                       # Experiment logs (not tracked)
```

## Results

### Same-Domain Evaluation

| Method | Backbone | LFW | CFP-FP | AgeDB-30 | Avg |
|--------|----------|-----|--------|----------|-----|
| Standard LoRA | DINOv2 ViT-S | 87.10% | ~85% | ~80% | ~84% |
| DA-LoRA (Ours) | DINOv2 ViT-S | **TBD** | **TBD** | **TBD** | **TBD** |

### Cross-Domain Evaluation

| Method | RGB→Thermal | Indoor→Outdoor |
|--------|-------------|----------------|
| Standard LoRA | ~70% | ~75% |
| DA-LoRA (Ours) | **TBD** | **TBD** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{adaptface2025,
  title={AdaptFace: Few-Shot Adaptation of Foundation Models for Cross-Domain Face Recognition via Domain-Aware LoRA},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## References

- [FRoundation: Are Foundation Models Ready for Face Recognition?](https://arxiv.org/abs/xxx)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [MAML: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)
- [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [timm](https://github.com/huggingface/pytorch-image-models) by Hugging Face
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
