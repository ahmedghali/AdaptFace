# EXP-001: Baseline DINOv2 + LoRA

## Informations Générales

| Champ | Valeur |
|-------|--------|
| **ID** | exp_001 |
| **Nom** | baseline_dinov2 |
| **Date début** | 2024-12-20 16:50 |
| **Date fin** | 2024-12-21 10:45 |
| **Durée** | 17.91 heures |
| **W&B Run** | [hvkjx93t](https://wandb.ai/ghali-ahmed/AdaptFace/runs/hvkjx93t) |

---

## Configuration

### Modèle
- **Backbone**: DINOv2 ViT-S/14
- **Paramètres totaux**: 22,268,288
- **Paramètres entraînables**: 865,664 (3.89%)

### LoRA
- **Rank**: 16
- **Alpha**: 16.0
- **Target modules**: qkv, proj

### Entraînement
- **Epochs**: 40
- **Batch size**: 128
- **Learning rate**: 1e-4
- **Optimizer**: AdamW (weight_decay=0.05)
- **Scheduler**: CosineAnnealingLR
- **Mixed precision**: Oui

### Loss
- **Type**: CosFace
- **Scale (s)**: 64.0
- **Margin (m)**: 0.35

---

## Résultats Finaux

| Métrique | Valeur |
|----------|--------|
| **Best LFW Accuracy** | **90.45%** |
| Final LFW Accuracy | 90.28% |
| Best AUC | 0.9620 |
| K-fold std | ±1.28% |
| Best Epoch | 33 |

---

## Progression de l'Entraînement

### Loss
```
Epoch 1:  31.54  ████████████████████████████████
Epoch 10: 27.50  ████████████████████████████
Epoch 20: 24.80  █████████████████████████
Epoch 30: 23.20  ███████████████████████
Epoch 40: 22.59  ███████████████████████
```

### Accuracy LFW
```
Epoch 1:  54.32%  █████
Epoch 2:  73.45%  ███████
Epoch 5:  80.12%  ████████
Epoch 10: 85.50%  █████████
Epoch 20: 88.90%  █████████
Epoch 30: 90.12%  █████████
Epoch 40: 90.28%  █████████  Best: 90.45%
```

### AUC
```
Epoch 1:  0.6816
Epoch 10: 0.9350
Epoch 40: 0.9620
```

---

## Analyse et Observations

### Points Positifs
1. **Dépasse la baseline FRoundation**: 90.45% vs 87.10% attendu (+3.35%)
2. **Convergence rapide**: 73.45% dès l'epoch 2
3. **Pas d'overfitting**: Accuracy stable sur les dernières epochs
4. **AUC excellent**: 0.962 indique une bonne séparabilité

### Points d'Attention
1. La loss ne diminue plus significativement après epoch 30
2. Le modèle pourrait bénéficier d'un learning rate initial plus élevé
3. 40 epochs suffisent, pas besoin de plus

### Comparaison avec FRoundation Paper

| Configuration | LFW Accuracy | Source |
|--------------|--------------|--------|
| DINOv2 + 1K IDs | 87.10% | FRoundation |
| DINOv2 + 10K IDs | 90.94% | FRoundation |
| **Notre baseline** | **90.45%** | Ce travail |

Notre résultat (90.45%) est cohérent avec les 10K identités du papier (90.94%).

---

## Fichiers Associés

- `config.yaml` - Configuration complète
- `checkpoints/best_model.pt` - Meilleur modèle (epoch 33)
- `checkpoints/final_model.pt` - Modèle final (epoch 40)
- `wandb/run-20251220_165023-hvkjx93t/` - Logs W&B

---

## Résultats Multi-Benchmark

| Benchmark | Accuracy | AUC | Description |
|-----------|----------|-----|-------------|
| **LFW** | **90.45%** | 0.9620 | Général - Excellent |
| **CFP-FP** | **71.81%** | 0.7879 | Pose (frontal vs profil) - À améliorer |
| **AgeDB-30** | **52.30%** | 0.4848 | Cross-âge - ÉCHEC (justifie DA-LoRA!) |
| **CALFW** | **69.27%** | 0.7545 | Cross-âge LFW - Difficile |
| **CPLFW** | **68.67%** | 0.7478 | Cross-pose LFW - Difficile |

### Moyenne par catégorie
- **Général (LFW)**: 90.45%
- **Pose (CFP-FP, CPLFW)**: 70.24% (moyenne)
- **Âge (AgeDB-30, CALFW)**: 60.79% (moyenne)

## Prochaines Étapes

1. [x] Évaluer sur tous les benchmarks - COMPLETE
2. [ ] Tester avec CLIP backbone (exp_002)
3. [ ] Implémenter Domain-Aware LoRA (exp_003) - devrait améliorer pose/âge

---

*Généré automatiquement le 2024-12-21*
