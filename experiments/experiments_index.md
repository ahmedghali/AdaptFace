# Index des Expériences AdaptFace

## Résumé Rapide

| # | Expérience | Backbone | LFW | CFP-FP | AgeDB | Durée | Date |
|---|------------|----------|-----|--------|-------|-------|------|
| 001 | Baseline LoRA | DINOv2 ViT-S | **90.45%** | - | - | 17.9h | 2024-12-20 |
| 002 | Baseline LoRA | CLIP ViT-B | - | - | - | - | Planifié |
| 003 | DA-LoRA 3 domains | DINOv2 ViT-S | - | - | - | - | Planifié |

---

## Détails par Expérience

### EXP-001: Baseline DINOv2 + LoRA
- **Dossier**: `exp_001_baseline_dinov2/`
- **Objectif**: Établir la baseline avec LoRA standard
- **Résultat**: 90.45% LFW (dépasse FRoundation 87.10%)
- **Statut**: ✅ Complete

### EXP-002: Baseline CLIP + LoRA (Planifié)
- **Dossier**: `exp_002_baseline_clip/`
- **Objectif**: Comparer CLIP vs DINOv2 comme backbone
- **Statut**: ⏳ À faire

### EXP-003: Domain-Aware LoRA (Planifié)
- **Dossier**: `exp_003_dalora_3domains/`
- **Objectif**: Innovation principale - LoRA avec modules domain-specific
- **Statut**: ⏳ À faire

---

## Graphique de Progression

```
Accuracy LFW (%)
    |
95% |                                    ← Objectif DA-LoRA
    |
90% |  ████ EXP-001 (90.45%)
    |
85% |  ---- FRoundation baseline (87.10%)
    |
80% |
    +----------------------------------------
        001   002   003   004   005
                Expérience #
```

---

*Dernière mise à jour: 2024-12-21*
