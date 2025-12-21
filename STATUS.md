# AdaptFace - Project Status

> **Quick reference pour ne jamais se perdre!**

---

## Où suis-je maintenant?

```
╔══════════════════════════════════════════════════════════════════╗
║  PHASE ACTUELLE: 3 - DA-LoRA IMPLEMENTATION COMPLETE ✅         ║
║  PROCHAINE ACTION: Lancer l'entraînement DA-LoRA                ║
║  DERNIÈRE ACTION: Code DA-LoRA implémenté et testé              ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Progression Globale

```
Phase 1: Setup          [████████████████████] 100% ✅
Phase 2: Baseline       [████████████████████] 100% ✅
Phase 3: DA-LoRA        [██████████░░░░░░░░░░]  50% ← CODE PRÊT, ENTRAÎNEMENT À LANCER
Phase 4: Experiments    [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
Phase 5: Ablation       [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
Phase 6: Paper          [░░░░░░░░░░░░░░░░░░░░]   0% ⏳
```

---

## Résultats EXP-001 (Baseline DINOv2 + LoRA)

| Benchmark | Accuracy | Type | Status |
|-----------|----------|------|--------|
| LFW | **90.45%** | Général | ✅ Excellent |
| CFP-FP | **71.81%** | Pose | ⚠️ À améliorer |
| AgeDB-30 | **52.30%** | Âge | ❌ Échec |
| CALFW | **69.27%** | Âge | ⚠️ Difficile |
| CPLFW | **68.67%** | Pose | ⚠️ Difficile |

**Conclusion**: Le baseline échoue sur pose/âge → justifie DA-LoRA!

---

## Prochaines Actions

### PRIORITÉ 1: Lancer l'entraînement DA-LoRA (EXP-003)
```bash
python train.py --backbone dinov2 --use-dalora --num-domains 3 --batch-size 128 --epochs 40 --wandb
```

### DA-LoRA Implementation Status
1. [x] Implémenter DomainAwareLoRA class
2. [x] Créer domain classifier
3. [x] Modifier train.py avec nouveaux arguments
4. [x] Tests passés (5/5)
5. [ ] Entraîner avec 3 domaines (général, pose, âge) ← À FAIRE

### PRIORITÉ 2: EXP-002 CLIP (optionnel)
```bash
python train.py --backbone clip --batch-size 64 --epochs 40 --wandb
```

---

## Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `src/models/da_lora.py` | **NOUVEAU** - Implementation DA-LoRA |
| `tests/test_dalora.py` | **NOUVEAU** - Tests DA-LoRA |
| `docs/GLOSSARY.md` | Explications des termes techniques |
| `experiments/` | Tracking des entraînements |
| `checkpoints/best_model.pt` | Meilleur modèle baseline |

---

## Commandes Utiles

```bash
# Tester DA-LoRA
python tests/test_dalora.py

# Entraîner avec DA-LoRA (NOUVELLE COMMANDE!)
python train.py --backbone dinov2 --use-dalora --num-domains 3 --batch-size 128 --epochs 40 --wandb

# Évaluer sur tous les benchmarks
python evaluate.py --checkpoint checkpoints/best_model.pt --benchmark all

# Entraînement standard (baseline)
python train.py --backbone dinov2 --batch-size 128 --epochs 40 --wandb
```

---

*Mis à jour: 2025-12-21*
