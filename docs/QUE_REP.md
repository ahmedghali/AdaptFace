# Questions & Réponses - AdaptFace

> Ce document explique les choix techniques du projet AdaptFace.

---

## 1. C'est quoi DINO et CLIP?

### DINOv2 (Meta AI, 2023)

**DINO** = **Di**stillation with **No** Labels

| Aspect | Description |
|--------|-------------|
| **Créateur** | Meta AI (Facebook) |
| **Type** | Self-supervised Vision Transformer |
| **Entraînement** | 142M images sans labels (auto-supervisé) |
| **Architecture** | ViT-S/14 (Small, patch 14x14) |
| **Paramètres** | 22M |
| **Force** | Excellentes features visuelles générales |

#### Qu'est-ce que "Self-supervised" (Auto-supervisé)?

**OUI, tu as bien compris!** Le modèle s'entraîne SANS labels d'identité.

```
Entraînement DINO:
┌─────────────────────────────────────────────────────────────┐
│  Image originale → [Augmentation 1] → Vue 1                │
│                  → [Augmentation 2] → Vue 2                │
│                                                             │
│  Objectif: Vue 1 et Vue 2 doivent avoir des features       │
│            SIMILAIRES (car c'est la même image!)           │
└─────────────────────────────────────────────────────────────┘

Le modèle apprend: "Quelles caractéristiques sont STABLES
malgré les transformations (rotation, crop, couleur...)?"
```

**Résultat**: DINO apprend à extraire des features visuelles robustes (contours, textures, formes) SANS savoir ce qu'est un visage, un chat, etc.

#### Features "Visuelles pures" - Qu'est-ce que ça veut dire?

```
DINO extrait des features de BAS NIVEAU:
┌─────────────────────────────────────────┐
│  Visage → DINO → [0.2, 0.8, 0.1, ...]  │
│                                         │
│  Ces nombres représentent:              │
│  - Contours du nez                      │
│  - Texture de la peau                   │
│  - Distance entre les yeux             │
│  - Forme des sourcils                   │
│  - Symétrie du visage                   │
│                                         │
│  = Caractéristiques GÉOMÉTRIQUES        │
│    et VISUELLES directes                │
└─────────────────────────────────────────┘
```

**Pourquoi pour la reconnaissance faciale?**
- DINO a appris sur 142M d'images diverses
- Il sait extraire des caractéristiques visuelles stables
- On ajoute LoRA/DA-LoRA pour le spécialiser sur les visages

---

### CLIP (OpenAI, 2021)

**CLIP** = **C**ontrastive **L**anguage-**I**mage **P**re-training

| Aspect | Description |
|--------|-------------|
| **Créateur** | OpenAI |
| **Type** | Vision-Language Transformer |
| **Entraînement** | 400M paires image-texte (contrastif) |
| **Architecture** | ViT-B/16 (Base, patch 16x16) |
| **Paramètres** | 86M (vision encoder) |
| **Force** | Features robustes aux variations |

#### Qu'est-ce que "Contrastif"?

```
Entraînement CLIP:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Paires POSITIVES (doivent être proches):                   │
│  [Photo de chat] ←→ "A photo of a cat"     ✓ Match!        │
│                                                             │
│  Paires NÉGATIVES (doivent être éloignées):                 │
│  [Photo de chat] ←→ "A photo of a dog"     ✗ Pas match!    │
│  [Photo de chat] ←→ "A red car"            ✗ Pas match!    │
│                                                             │
│  Objectif: Rapprocher image-texte qui correspondent,        │
│            Éloigner ceux qui ne correspondent pas           │
└─────────────────────────────────────────────────────────────┘
```

**IMPORTANT pour notre projet**: On utilise SEULEMENT la partie vision de CLIP (pas le texte!). On jette le text encoder.

#### Features "Sémantiques" - Qu'est-ce que ça veut dire?

```
CLIP extrait des features de HAUT NIVEAU:
┌─────────────────────────────────────────┐
│  Visage → CLIP → [0.2, 0.8, 0.1, ...]  │
│                                         │
│  Ces nombres représentent:              │
│  - "Personne âgée" vs "Jeune"          │
│  - "Expression souriante"               │
│  - "Visage de profil"                   │
│  - "Éclairage studio"                   │
│                                         │
│  = Concepts de HAUT NIVEAU              │
│    (car entraîné avec du texte)         │
└─────────────────────────────────────────┘
```

#### Pourquoi CLIP pour la reconnaissance faciale?

**ATTENTION**: On n'utilise PAS le texte pour reconnaître! Voici pourquoi CLIP peut aider:

```
Le vision encoder de CLIP a appris des features ROBUSTES:
- Il a vu "young woman smiling" et "old man serious"
- Donc il a appris à distinguer âge, expression, pose
- Ces features peuvent aider pour les cas difficiles

Pour nous:
- On prend JUSTE le vision encoder
- On ajoute LoRA pour le fine-tuner sur les visages
- On utilise ses features robustes, PAS le texte
```

---

## 2. C'est quoi LoRA?

**LoRA** = **Lo**w-**R**ank **A**daptation (Microsoft, 2021)

### Le problème que LoRA résout

```
Fine-tuning COMPLET (méthode classique):
┌─────────────────────────────────────────────────────────────┐
│  DINO a 22 millions de paramètres                          │
│  → Il faut stocker 22M de gradients en mémoire             │
│  → Il faut modifier 22M de poids                           │
│  → TRÈS COÛTEUX en mémoire GPU!                            │
└─────────────────────────────────────────────────────────────┘
```

### L'idée de LoRA: Factorisation matricielle

**Question**: Pourquoi LoRA n'a pas la même taille que W?

**Réponse**: Grâce à la factorisation LOW-RANK (rang faible)!

```
Prenons une couche linéaire de DINO:
┌─────────────────────────────────────────────────────────────┐
│  W = matrice [384 × 384] = 147,456 paramètres              │
│                                                             │
│  Fine-tuning classique:                                     │
│  ΔW = matrice [384 × 384] = 147,456 paramètres à entraîner │
│                                                             │
│  LoRA avec rank=16:                                         │
│  A = matrice [16 × 384]  = 6,144 paramètres                │
│  B = matrice [384 × 16]  = 6,144 paramètres                │
│  Total LoRA = 12,288 paramètres (8% de W!)                 │
└─────────────────────────────────────────────────────────────┘
```

### Visualisation des dimensions

```
                    Fine-tuning classique
                    ┌─────────────────┐
                    │                 │
                    │   ΔW [384×384]  │  = 147,456 params
                    │                 │
                    └─────────────────┘

                    LoRA (rank=16)
            ┌───┐
            │   │  B [384×16]
            │   │  = 6,144 params
            │   │
            └───┘
               ×
        ┌─────────────────┐
        │  A [16×384]     │ = 6,144 params
        └─────────────────┘
               ↓
        ┌─────────────────┐
        │  B × A [384×384]│  = Même forme que W!
        └─────────────────┘

Total LoRA: 6,144 + 6,144 = 12,288 params (vs 147,456)
```

### L'équation expliquée

```
W' = W + B × A

Où:
- W  = poids originaux [384 × 384] → GELÉS (on ne touche pas!)
- A  = petite matrice  [16 × 384]  → ENTRAÎNABLE
- B  = petite matrice  [384 × 16]  → ENTRAÎNABLE
- B × A = produit matriciel [384 × 384] → Même taille que W!

Le produit B × A donne une matrice de MÊME TAILLE que W,
mais on n'entraîne que A et B (beaucoup plus petits).
```

### D'où vient le "1-5%" de paramètres?

**Calcul pour DINOv2 ViT-S/14:**

```
DINO a 12 blocs Transformer, chaque bloc a:
- qkv (query, key, value): [384 × 1152] = 442,368 params
- proj (projection):       [384 × 384]  = 147,456 params

On applique LoRA à qkv et proj dans chaque bloc:

LoRA pour qkv (rank=16):
- A: [16 × 384] = 6,144
- B: [1152 × 16] = 18,432
- Total: 24,576 params par bloc

LoRA pour proj (rank=16):
- A: [16 × 384] = 6,144
- B: [384 × 16] = 6,144
- Total: 12,288 params par bloc

Total LoRA par bloc: 24,576 + 12,288 = 36,864
Total LoRA (12 blocs): 36,864 × 12 = 442,368 params

Pourcentage: 442,368 / 22,000,000 = 2.0%
```

**Donc ~2% des paramètres sont entraînés, pas 100%!**

### Résumé LoRA

| Aspect | Fine-tuning complet | LoRA rank=16 |
|--------|---------------------|--------------|
| Params entraînés | 22M (100%) | ~442K (2%) |
| Mémoire GPU | ~8 GB | ~2 GB |
| Backbone modifié | Oui | Non (gelé) |
| Temps entraînement | Lent | Rapide |

---

## 3. C'est quoi DA-LoRA? (Notre contribution!)

**DA-LoRA** = **D**omain-**A**ware **LoRA**

### Problème que ça résout

Le LoRA standard utilise UNE SEULE adaptation pour TOUTES les images. Mais les visages varient selon:
- **Pose**: frontal, profil, 3/4
- **Âge**: jeune, adulte, âgé
- **Éclairage**: studio, extérieur, nuit

### Notre solution: Multiple LoRA + Domain Classifier

```
Standard LoRA:   W' = W + B × A                    (1 seule adaptation)
DA-LoRA:         W' = W + Σ(wₖ × Bₖ × Aₖ)          (K adaptations pondérées)

Où:
- K = nombre de domaines (3 dans notre cas)
- wₖ = poids du domaine k (prédit par le classifier)
- (Aₖ, Bₖ) = matrices LoRA spécifiques au domaine k
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DA-LoRA                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image → Backbone → Features → Domain Classifier            │
│              │                      │                       │
│              │                      ▼                       │
│              │               [w₁, w₂, w₃]                   │
│              │                      │                       │
│              ▼                      ▼                       │
│         ┌─────────────────────────────────┐                 │
│         │  w₁ × (B₁ × A₁)  ← Domaine 1    │                 │
│         │+ w₂ × (B₂ × A₂)  ← Domaine 2    │                 │
│         │+ w₃ × (B₃ × A₃)  ← Domaine 3    │                 │
│         └─────────────────────────────────┘                 │
│                      │                                      │
│                      ▼                                      │
│              Embedding final                                │
└─────────────────────────────────────────────────────────────┘
```

### Avantages de DA-LoRA

| Avantage | Explication |
|----------|-------------|
| **Spécialisation** | Chaque domaine a son adaptation |
| **Dynamique** | Poids calculés pour chaque image |
| **Robustesse** | Meilleur sur pose/âge difficiles |
| **Interprétable** | On voit quel domaine est activé |

---

## 4. Pourquoi DINO et CLIP comme backbones?

### Comparaison

| Critère | DINOv2 | CLIP |
|---------|--------|------|
| **Entraînement original** | Images seules | Images + texte |
| **Features extraites** | Géométrie, texture | Attributs, variations |
| **Taille modèle** | 22M (petit, rapide) | 86M (moyen, plus lent) |
| **Force pour visages** | Détails fins du visage | Robustesse aux variations |

### Pourquoi ces deux-là?

```
┌─────────────────────────────────────────────────────────────┐
│  Nos critères de sélection:                                 │
│                                                             │
│  1. Pré-entraînés sur BEAUCOUP de données (pas de scratch) │
│  2. Architecture ViT (Vision Transformer) - état de l'art  │
│  3. Features de qualité pour le fine-tuning                │
│  4. Compatible avec notre GPU (pas trop gros)              │
└─────────────────────────────────────────────────────────────┘
```

1. **DINOv2 (choix principal)**
   - 22M paramètres = rapide à entraîner
   - Excellentes features visuelles
   - Pré-entraîné sur 142M images
   - Idéal pour notre RTX 3060/4070

2. **CLIP (alternative pour comparaison)**
   - Approche différente (vision + langage)
   - Peut capturer des attributs différents
   - Test si ça améliore les cas difficiles (âge, pose)

### Ce qu'on NE choisit PAS

| Backbone | Pourquoi pas? |
|----------|---------------|
| ResNet-50 | Architecture ancienne (2015), moins bon que ViT |
| VGGFace | Spécifique visages mais architecture dépassée |
| Entraînement from scratch | Trop long, pas assez de données |

---

## 5. Pourquoi pas CLIP + DA-LoRA?

**Excellente question!** On PEUT et on DEVRAIT le faire!

### Plan d'expériences complet

| Exp | Backbone | Adaptation | Priorité | Status |
|-----|----------|------------|----------|--------|
| EXP-001 | DINOv2 | LoRA | Baseline | ✅ Terminé |
| EXP-002 | CLIP | LoRA | Alternative | ⏳ À faire |
| **EXP-003** | **DINOv2** | **DA-LoRA** | **Principal** | 🔄 En cours |
| **EXP-004** | **CLIP** | **DA-LoRA** | **Nouveau!** | ⏳ À planifier |

### Pourquoi j'ai dit seulement CLIP + LoRA?

C'était une simplification. La matrice complète des expériences est:

```
                    LoRA          DA-LoRA
              ┌─────────────┬─────────────┐
    DINOv2    │  EXP-001    │  EXP-003    │
              │  (baseline) │  (principal)│
              ├─────────────┼─────────────┤
    CLIP      │  EXP-002    │  EXP-004    │
              │(alternative)│  (nouveau!) │
              └─────────────┴─────────────┘
```

### Ordre recommandé

1. **EXP-001** ✅ DINOv2 + LoRA (baseline fait)
2. **EXP-003** 🔄 DINOv2 + DA-LoRA (en cours)
3. **EXP-004** ⏳ CLIP + DA-LoRA (après EXP-003)
4. **EXP-002** ⏳ CLIP + LoRA (optionnel, pour comparaison)

### Commande pour CLIP + DA-LoRA

```bash
python train.py --backbone clip --use-dalora --num-domains 3 --batch-size 64 --epochs 40 --wandb
```

---

## Résumé

```
┌────────────────────────────────────────────────────────────┐
│                    AdaptFace Architecture                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Backbones (features de base):                            │
│   ├── DINOv2: Auto-supervisé, features visuelles          │
│   └── CLIP: Vision-langage, features sémantiques          │
│                                                            │
│   Adaptations (fine-tuning efficace):                      │
│   ├── LoRA: Une adaptation pour tout                       │
│   └── DA-LoRA: Adaptations spécifiques par domaine        │
│                                                            │
│   Notre contribution = DA-LoRA                             │
│   → Meilleur sur pose/âge difficiles                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

*Document créé le 2025-12-22*
