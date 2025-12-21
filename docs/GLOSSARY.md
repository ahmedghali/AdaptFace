# Glossaire AdaptFace

> Explication des termes techniques du projet

---

## Termes Principaux

### LoRA (Low-Rank Adaptation)
```
Technique pour adapter un gros modèle avec très peu de paramètres.

AVANT (Fine-tuning classique):
┌─────────────────────────────────┐
│  Modèle (22M paramètres)        │  ← Tout est modifié = lent, coûteux
│  Tous les poids changent        │
└─────────────────────────────────┘

APRÈS (LoRA):
┌─────────────────────────────────┐
│  Modèle (22M paramètres)        │  ← Gelé (frozen)
│  [Frozen - ne change pas]       │
│                                 │
│  + Petit module LoRA (800K)     │  ← Seul ça change = rapide, efficace
│    Matrices A et B de rang r    │
└─────────────────────────────────┘

Avantages:
- 3.89% des paramètres entraînables (au lieu de 100%)
- Entraînement plus rapide
- Moins de mémoire GPU
- Évite l'overfitting
```

---

### DA-LoRA (Domain-Aware LoRA) - NOTRE INNOVATION!
```
LoRA amélioré avec des modules spécifiques pour chaque "domaine".

Domaines = conditions différentes:
- Pose (frontal vs profil)
- Âge (jeune vs vieux)
- Éclairage (intérieur vs extérieur)

ARCHITECTURE:
                    ┌─────────────────┐
                    │  Image d'entrée │
                    └────────┬────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────┐
│              Domain Classifier                      │
│  "Cette image est: 60% pose, 30% âge, 10% général" │
└────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ LoRA    │    │ LoRA    │    │ LoRA    │
        │ Général │    │ Pose    │    │ Âge     │
        └────┬────┘    └────┬────┘    └────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Somme pondérée  │
                    │ des sorties     │
                    └─────────────────┘

Avantage: Le modèle s'adapte automatiquement au type d'image!
```

---

### Backbone (Colonne vertébrale)
```
Le modèle de base pré-entraîné qui extrait les features des images.

Dans ce projet, 2 options:

┌─────────────────────────────────────────────────────────┐
│  DINOv2 (Facebook/Meta)                                 │
│  ─────────────────────                                  │
│  • Vision Transformer Small (ViT-S/14)                  │
│  • 22M paramètres                                       │
│  • Pré-entraîné sur millions d'images (self-supervised) │
│  • Très bon pour comprendre la structure des images     │
│  • Résultat: 90.45% sur LFW ✅                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  CLIP (OpenAI)                                          │
│  ────────────                                           │
│  • Vision Transformer Base (ViT-B/16)                   │
│  • 86M paramètres                                       │
│  • Pré-entraîné sur images + texte                      │
│  • Comprend le lien entre images et descriptions        │
│  • Potentiellement meilleur mais plus lourd             │
└─────────────────────────────────────────────────────────┘
```

---

### CosFace / ArcFace (Fonctions de perte)
```
Techniques pour apprendre des embeddings discriminants.

PROBLÈME:
  Comment séparer les visages de différentes personnes?

SOLUTION - Marge angulaire:

  Espace des embeddings (simplifié en 2D):

       Personne A          Personne B
           ●                   ●
          /│\                 /│\
         / │ \               / │ \
        /  │  \             /  │  \
       /   │   \           /   │   \
      ────────────────────────────────

  CosFace: Ajoute une marge m au cosinus
  Formule: cos(θ) - m

  Plus m est grand, plus les classes sont séparées.

  Nos paramètres:
  - s (scale) = 64.0  → Amplifie les différences
  - m (margin) = 0.35 → Force de séparation
```

---

### Benchmarks (Jeux de test)
```
Datasets pour évaluer la performance du modèle.

┌────────────┬─────────┬──────────────────────────────────────┐
│ Benchmark  │ Paires  │ Ce qu'il teste                       │
├────────────┼─────────┼──────────────────────────────────────┤
│ LFW        │ 6,000   │ Conditions générales (varié)         │
│ CFP-FP     │ 7,000   │ Frontal vs Profil (90° rotation)     │
│ AgeDB-30   │ 6,000   │ Même personne, 30 ans d'écart        │
│ CALFW      │ 6,000   │ Variations d'âge (Cross-Age LFW)     │
│ CPLFW      │ 6,000   │ Variations de pose (Cross-Pose LFW)  │
└────────────┴─────────┴──────────────────────────────────────┘

Comment ça marche:
1. Prendre 2 images
2. Le modèle dit "même personne" ou "personnes différentes"
3. Comparer avec la vérité → Accuracy
```

---

### Embeddings (Représentations)
```
Vecteur numérique représentant un visage.

Image (224x224 pixels)  →  Modèle  →  Embedding (512 nombres)
       150,528 valeurs                      512 valeurs

Propriétés:
- 2 images de la MÊME personne → embeddings SIMILAIRES
- 2 images de personnes DIFFÉRENTES → embeddings DIFFÉRENTS

Mesure de similarité: Cosinus
- cos = 1.0  → Identiques
- cos = 0.0  → Aucun rapport
- cos = -1.0 → Opposés
```

---

### Mixed Precision (Précision mixte)
```
Technique pour accélérer l'entraînement.

AVANT: Tous les calculs en float32 (32 bits)
APRÈS: Mélange de float16 (16 bits) et float32

Avantages:
- 2x plus rapide
- 2x moins de mémoire GPU
- Qualité identique

Dans notre code:
  with torch.amp.autocast('cuda'):
      output = model(images)
```

---

### K-fold Cross-Validation
```
Méthode pour évaluer la robustesse d'un modèle.

Données de test (6000 paires):
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ F1   │ F2   │ F3   │ F4   │ F5   │ F6   │ F7   │ F8   │ F9   │ F10  │
│ 600  │ 600  │ 600  │ 600  │ 600  │ 600  │ 600  │ 600  │ 600  │ 600  │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

Fold 1: Tester sur F1, trouver seuil avec F2-F10
Fold 2: Tester sur F2, trouver seuil avec F1,F3-F10
...
Fold 10: Tester sur F10, trouver seuil avec F1-F9

Résultat: Accuracy moyenne ± écart-type
Exemple: 90.45% (±1.28%)
```

---

### AUC (Area Under Curve)
```
Mesure de la qualité du classifieur (0 à 1).

Interprétation:
- AUC = 1.0  → Parfait (sépare tout)
- AUC = 0.5  → Random (comme pile ou face)
- AUC < 0.5  → Pire que random (labels inversés?)

Nos résultats:
- LFW:     0.9620 → Excellent
- CFP-FP:  0.7879 → Bon
- AgeDB:   0.4848 → Problème! (< 0.5)
```

---

### TAR@FAR (True Accept Rate @ False Accept Rate)
```
Métrique de sécurité.

FAR (False Accept Rate): % d'imposteurs acceptés par erreur
TAR (True Accept Rate): % de vraies personnes acceptées

TAR@FAR=0.1% signifie:
"Si j'accepte 1 imposteur sur 1000, combien de vraies personnes j'accepte?"

Plus TAR est haut à FAR bas = meilleur modèle.
```

---

## Abréviations

| Abréviation | Signification |
|-------------|---------------|
| LoRA | Low-Rank Adaptation |
| DA-LoRA | Domain-Aware LoRA |
| ViT | Vision Transformer |
| LFW | Labeled Faces in the Wild |
| CFP-FP | Celebrities Frontal-Profile |
| AUC | Area Under Curve |
| TAR | True Accept Rate |
| FAR | False Accept Rate |
| W&B | Weights & Biases |

---

*Dernière mise à jour: 2025-12-21*
