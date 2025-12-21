# Literature Review Notes - AdaptFace Project

This document organizes the literature review for the DA-LoRA project. Complete each section as you read the papers.

---

## Progress Tracker

| Paper       | Status          | Priority |
|-------------|-----------------|----------|
| FRoundation | [ ] Not Started | HIGH     |
| LoRA        | [ ] Not Started | HIGH     |
| MAML        | [ ] Not Started | HIGH     |
| ArcFace     | [ ] Not Started | HIGH     |
| CosFace     | [ ] Not Started | HIGH     |
| DINOv2      | [ ] Not Started | HIGH     |
| CLIP        | [ ] Not Started | MEDIUM   |
| DANN        | [ ] Not Started | MEDIUM   |
| ADDA        | [ ] Not Started | LOW      |

---

## 1. FRoundation Paper (Foundation Models in Face Recognition)

**Title:** FRoundation: Are Foundation Models Ready for Face Recognition?
**Link:** https://arxiv.org/abs/2407.09569
**Year:** 2024

### Key Questions to Answer:
- [ ] What foundation models were tested for face recognition?
- [ ] What was the evaluation protocol used?
- [ ] What were the baseline results for DINOv2 and CLIP?
- [ ] What fine-tuning approaches did they try?
- [ ] What are the limitations identified?

### Summary:
```
[Write 2-3 paragraph summary here]
```

### Key Results (copy from paper):
| Model | LFW | CFP-FP | AgeDB-30 |
|-------|-----|--------|----------|
| DINOv2 ViT-S + 1K | | | |
| DINOv2 ViT-S + 10K | | | |
| CLIP ViT-B + 1K | | | |

### Implementation Details for Our Project:
- Data preprocessing:
- Hyperparameters:
- Training protocol:

### Notes:
```
[Your annotations here]
```

---

## 2. LoRA Paper (Low-Rank Adaptation)

**Title:** LoRA: Low-Rank Adaptation of Large Language Models
**Authors:** Hu et al.
**Link:** https://arxiv.org/abs/2106.09685
**Year:** 2021

### Key Questions to Answer:
- [ ] What is the mathematical formulation of LoRA?
- [ ] Why does low-rank adaptation work?
- [ ] What is the rank (r) and how does it affect performance?
- [ ] Which layers should LoRA be applied to?
- [ ] What is the initialization strategy?

### Mathematical Formulation:
```
W = W_0 + BA where:
- W_0: frozen pre-trained weights (d x k)
- B: trainable matrix (d x r)
- A: trainable matrix (r x k)
- r << min(d, k): the rank
```

### Key Insights:
-

### Implementation Checklist:
- [ ] Matrix A: Kaiming initialization
- [ ] Matrix B: Zero initialization
- [ ] Apply to: Wq and Wv in attention layers
- [ ] Typical rank: 4, 8, 16, 32

### Notes:
```
[Your annotations here]
```

---

## 3. MAML Paper (Meta-Learning)

**Title:** Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
**Authors:** Finn et al.
**Link:** https://arxiv.org/abs/1703.03400
**Year:** 2017

### Key Questions to Answer:
- [ ] What is the MAML algorithm (inner loop, outer loop)?
- [ ] How many inner loop steps are typically used?
- [ ] What is the meta-learning rate vs task learning rate?
- [ ] How do we define "tasks" for face recognition domains?

### Algorithm Pseudocode:
```
For each meta-iteration:
    Sample batch of tasks T_i
    For each task T_i:
        Compute adapted parameters: θ'_i = θ - α∇L_T_i(θ)  # inner loop
    Update θ = θ - β∇Σ_i L_T_i(θ'_i)  # outer loop
```

### Implementation Considerations for DA-LoRA:
- Task definition: Each domain = one task
- Inner loop steps: 5 (as in project spec)
- Support set / Query set design:

### Notes:
```
[Your annotations here]
```

---

## 4. ArcFace Paper

**Title:** ArcFace: Additive Angular Margin Loss for Deep Face Recognition
**Authors:** Deng et al.
**Link:** https://arxiv.org/abs/1801.07698
**Year:** 2019

### Key Questions to Answer:
- [ ] What is the angular margin formulation?
- [ ] How does it differ from softmax loss?
- [ ] What are the hyperparameters (s, m)?
- [ ] Why is angular margin effective for face recognition?

### Loss Formulation:
```
L = -log(exp(s*cos(θ_y + m)) / (exp(s*cos(θ_y + m)) + Σ exp(s*cos(θ_j))))

where:
- s: scale factor (typically 64)
- m: angular margin (typically 0.5)
- θ_y: angle between feature and class center
```

### Notes:
```
[Your annotations here]
```

---

## 5. CosFace Paper

**Title:** CosFace: Large Margin Cosine Loss for Deep Face Recognition
**Authors:** Wang et al.
**Link:** https://arxiv.org/abs/1801.09414
**Year:** 2018

### Key Questions to Answer:
- [ ] How does cosine margin differ from angular margin?
- [ ] What are the recommended hyperparameters?
- [ ] Why did we choose CosFace over ArcFace for this project?

### Loss Formulation:
```
L = -log(exp(s*(cos(θ_y) - m)) / (exp(s*(cos(θ_y) - m)) + Σ exp(s*cos(θ_j))))

where:
- s: scale factor (64.0 in our project)
- m: cosine margin (0.35 in our project)
```

### Notes:
```
[Your annotations here]
```

---

## 6. DINOv2 Paper

**Title:** DINOv2: Learning Robust Visual Features without Supervision
**Authors:** Oquab et al. (Meta AI)
**Link:** https://arxiv.org/abs/2304.07193
**Year:** 2023

### Key Questions to Answer:
- [ ] What is self-distillation in DINO?
- [ ] What makes DINOv2 different from DINOv1?
- [ ] What pre-training data was used?
- [ ] What are the available model sizes?
- [ ] Why is DINOv2 good for face recognition?

### Model Variants:
| Model | Params | Patch Size |
|-------|--------|------------|
| ViT-S/14 | 22M | 14x14 |
| ViT-B/14 | 86M | 14x14 |
| ViT-L/14 | 307M | 14x14 |
| ViT-g/14 | 1.1B | 14x14 |

### Key Features for Face Recognition:
-

### Notes:
```
[Your annotations here]
```

---

## 7. CLIP Paper

**Title:** Learning Transferable Visual Models From Natural Language Supervision
**Authors:** Radford et al. (OpenAI)
**Link:** https://arxiv.org/abs/2103.00020
**Year:** 2021

### Key Questions to Answer:
- [ ] What is contrastive language-image pre-training?
- [ ] How large is the training dataset?
- [ ] What visual encoder architectures are available?
- [ ] How does CLIP compare to DINOv2 for face recognition?

### Model Variants:
| Model | Vision Encoder | Params |
|-------|---------------|--------|
| ViT-B/32 | ViT-Base | 86M |
| ViT-B/16 | ViT-Base | 86M |
| ViT-L/14 | ViT-Large | 307M |

### Notes:
```
[Your annotations here]
```

---

## 8. DANN Paper (Domain Adaptation)

**Title:** Domain-Adversarial Training of Neural Networks
**Authors:** Ganin et al.
**Link:** https://arxiv.org/abs/1505.07818
**Year:** 2016

### Key Questions to Answer:
- [ ] What is the domain adversarial approach?
- [ ] What is the gradient reversal layer?
- [ ] How does it encourage domain-invariant features?

### Architecture:
```
Input -> Feature Extractor -> |-> Label Predictor
                              |-> Domain Classifier (with GRL)
```

### Relevance to DA-LoRA:
-

### Notes:
```
[Your annotations here]
```

---

## 9. ADDA Paper (Domain Adaptation)

**Title:** Adversarial Discriminative Domain Adaptation
**Authors:** Tzeng et al.
**Link:** https://arxiv.org/abs/1702.05464
**Year:** 2017

### Key Questions to Answer:
- [ ] How does ADDA differ from DANN?
- [ ] What is the two-stage training process?

### Notes:
```
[Your annotations here]
```

---

## Key Insights Summary

### What makes DA-LoRA novel?

1. **Compared to standard LoRA:**
   -

2. **Compared to DANN/ADDA:**
   -

3. **Compared to FRoundation approach:**
   -

### Research Gap Identified:
```
[Summarize the gap that DA-LoRA addresses]
```

### Our Contributions:
1.
2.
3.

---

## Annotated Bibliography (for paper)

### Face Recognition
-

### Domain Adaptation
-

### Parameter-Efficient Fine-Tuning
-

### Meta-Learning
-

---

*Last Updated: [Date]*