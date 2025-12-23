# Index des Expériences AdaptFace

## Résumé Rapide

| # | Expérience | Backbone | LFW | AUC | Durée | Date | Statut |
|---|------------|----------|-----|-----|-------|------|--------|
| 001 | Baseline LoRA | DINOv2 ViT-S | **90.45%** | 0.962 | 17.9h | 2025-12-20 | ✅ Complete |
| 002 | Baseline LoRA | CLIP ViT-B | - | - | - | - | ⏳ Optionnel |
| 003 | **DA-LoRA** | DINOv2 ViT-S | **89.50%** | 0.957 | 37.5h | 2025-12-23 | ✅ **COMPLETE** |

---

## Résultats Multi-Benchmark

### EXP-001 (Baseline)

| Benchmark | Accuracy | AUC | Type |
|-----------|----------|-----|------|
| LFW | **90.45%** | 0.962 | Général |
| CFP-FP | **71.81%** | 0.788 | Pose |
| AgeDB-30 | **52.30%** | 0.485 | Âge |
| CALFW | **69.27%** | 0.755 | Âge |
| CPLFW | **68.67%** | 0.748 | Pose |

### EXP-003 (DA-LoRA) ✅ Évalué

| Benchmark | Accuracy | AUC | vs Baseline | Type |
|-----------|----------|-----|-------------|------|
| LFW | **89.50%** | 0.957 | -0.95% | Général |
| CFP-FP | **72.73%** | 0.796 | **+0.92%** ✅ | Pose |
| AgeDB-30 | **54.17%** | 0.514 | **+1.87%** ✅ | Âge |
| CALFW | **68.45%** | 0.750 | -0.82% | Âge |
| CPLFW | **68.82%** | 0.748 | **+0.15%** ✅ | Pose |

**Résumé**: DA-LoRA améliore les domaines difficiles (pose/âge) avec un léger trade-off sur LFW.

---

## Détails par Expérience

### EXP-001: Baseline DINOv2 + LoRA ✅
- **Dossier**: `exp_001_baseline_dinov2/`
- **Objectif**: Établir la baseline avec LoRA standard
- **Résultat**: 90.45% LFW (dépasse FRoundation 87.10%)
- **Trainable params**: 865,664 (3.89%)
- **Statut**: ✅ Complete

### EXP-002: Baseline CLIP + LoRA (Optionnel)
- **Dossier**: `exp_002_baseline_clip/`
- **Objectif**: Comparer CLIP vs DINOv2 comme backbone
- **Statut**: ⏳ Optionnel

### EXP-003: Domain-Aware LoRA ✅ COMPLETE + ÉVALUÉ
- **Dossier**: `exp_003_dalora_dinov2/`
- **Objectif**: Innovation principale - LoRA avec modules domain-specific
- **Configuration**:
  - Num domains: 3
  - Domain momentum: 0.9 (EMA)
  - Single-pass optimization
- **Résultats**:
  - LFW: 89.50% | CFP-FP: 72.73% | AgeDB-30: 54.17%
  - CALFW: 68.45% | CPLFW: 68.82%
- **Améliorations vs Baseline**: CFP-FP +0.92%, AgeDB-30 +1.87%, CPLFW +0.15%
- **Trainable params**: 1,849,731 (7.96%)
- **Durée**: 37.46 heures
- **Statut**: ✅ **COMPLETE + ÉVALUÉ**

---

## Graphique de Progression

```
Accuracy LFW (%)
    |
95% |                                    ← Objectif final
    |
90% |  ████ EXP-001 (90.45%)  ███ EXP-003 (89.50%)
    |
85% |  ---- FRoundation baseline (87.10%)
    |
80% |
    +----------------------------------------
        001   002   003   004   005
                Expérience #
```

---

## Fichiers Importants

| Expérience | Checkpoint | Config | Results |
|------------|------------|--------|---------|
| EXP-001 | `checkpoints/checkpoint_epoch_33.pt` | `config.yaml` | `results.md` |
| EXP-003 | `checkpoints/best_model.pt` | `config.yaml` | `results.md` |

---

*Dernière mise à jour: 2025-12-23*
