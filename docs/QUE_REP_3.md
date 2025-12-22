# Questions & Réponses - Partie 3 (Questions Avancées)

> Explications approfondies sur les Transformers, l'entraînement, et HuggingFace.

---

## 1. Google Colab vs Mon PC - Gain de temps?

### Comparaison des GPUs

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPARAISON GPU                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TON PC (RTX 3060/4070):                                       │
│  ├─ VRAM: 8-12 GB                                              │
│  ├─ Performance: ~15-20 TFLOPS                                 │
│  └─ Disponibilité: 24/7, pas de limite                         │
│                                                                 │
│  GOOGLE COLAB (Gratuit):                                       │
│  ├─ GPU: T4 (16 GB VRAM)                                       │
│  ├─ Performance: ~8 TFLOPS (plus LENT que ton PC!)            │
│  ├─ Limite: ~12h puis déconnexion                              │
│  └─ File d'attente: peut être indisponible                     │
│                                                                 │
│  GOOGLE COLAB PRO ($10/mois):                                  │
│  ├─ GPU: V100 ou A100 (16-40 GB VRAM)                         │
│  ├─ Performance: ~30-80 TFLOPS                                 │
│  ├─ Limite: ~24h                                               │
│  └─ Priorité d'accès                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Estimation du temps

| Configuration | Temps estimé (40 epochs) |
|---------------|--------------------------|
| Ton PC (RTX 3060) | ~36 heures |
| Colab Gratuit (T4) | ~45-50 heures (plus LENT + déconnexions!) |
| Colab Pro (V100) | ~18-20 heures |
| Colab Pro (A100) | ~10-12 heures |

### Ma recommandation

```
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMANDATION:                                                │
│                                                                 │
│  1. GARDE ton PC pour l'entraînement                           │
│     - Pas de déconnexion                                        │
│     - Pas de limite de temps                                    │
│     - Tu peux dormir pendant que ça tourne                     │
│                                                                 │
│  2. Utilise Colab SEULEMENT pour:                              │
│     - Tests rapides                                             │
│     - Debugging                                                 │
│     - Si tu veux tester A100 (Colab Pro)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. C'est quoi Self-Attention? (Explication simple)

### Le problème que Self-Attention résout

```
Imagine une phrase: "Le chat dort sur le canapé"

Question: Comment le modèle sait que "dort" est lié à "chat"?

SANS attention: Chaque mot est traité indépendamment
  → Le modèle ne comprend pas les relations

AVEC attention: Chaque mot "regarde" tous les autres mots
  → Le modèle comprend que "dort" est l'action du "chat"
```

### Pour les images (notre cas)

```
Une image 224×224 est découpée en PATCHES (morceaux):

┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │     16×16 patches
├───┼───┼───┼───┤     = 256 patches pour 224×224 (16×16)
│ 5 │ 6 │ 7 │ 8 │     (avec patch size 14)
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘

Self-Attention permet à CHAQUE patch de "regarder"
TOUS les autres patches pour comprendre l'image globale.
```

### Exemple concret pour un visage

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Patch "œil gauche" regarde:                                  │
│   ├─ Patch "œil droit" → "Ah, il y a symétrie!"               │
│   ├─ Patch "nez" → "Je suis au-dessus du nez"                 │
│   ├─ Patch "sourcil" → "Mon sourcil est juste au-dessus"      │
│   └─ Patch "bouche" → "La bouche est plus bas"                │
│                                                                 │
│   Résultat: Le patch "œil gauche" comprend sa POSITION         │
│   et ses RELATIONS avec le reste du visage!                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Visualisation du calcul

```
Self-Attention = "Qui doit regarder qui, et combien?"

Entrée: 256 patches, chaque patch = vecteur de 384 dims

Étape 1: Calculer les "scores d'attention"
┌─────────────────────────────────────────┐
│  Patch 1 regarde Patch 1: score = 0.8   │
│  Patch 1 regarde Patch 2: score = 0.1   │
│  Patch 1 regarde Patch 3: score = 0.05  │
│  ...                                     │
│  (score élevé = "je dois faire attention │
│   à ce patch!")                          │
└─────────────────────────────────────────┘

Étape 2: Combiner selon les scores
┌─────────────────────────────────────────┐
│  Nouvelle représentation du Patch 1 =   │
│    0.8 × Patch1 + 0.1 × Patch2 + ...   │
│                                         │
│  = Mélange intelligent de tous les     │
│    patches, pondéré par l'importance   │
└─────────────────────────────────────────┘
```

---

## 3. C'est quoi MLP (Feed-Forward)?

### Comparaison simple

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SELF-ATTENTION = Communication ENTRE patches                   │
│  "Les patches se parlent entre eux"                            │
│                                                                 │
│  MLP = Traitement INDIVIDUEL de chaque patch                   │
│  "Chaque patch réfléchit seul"                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Structure du MLP

```
MLP = 2 couches linéaires avec une activation

Entrée [384] → Linear1 [384→1536] → GELU → Linear2 [1536→384] → Sortie [384]
                     ↑                              ↑
                 Expansion (×4)                 Réduction
                "Réfléchir plus"            "Résumer"
```

### Analogie

```
Self-Attention = Réunion de groupe
  "Tout le monde partage ses idées"

MLP = Travail individuel
  "Chacun digère les informations reçues"

Les deux sont nécessaires!
```

### Pourquoi on met LoRA sur l'Attention et pas le MLP?

```
Raison: L'attention capture les RELATIONS (plus important!)

Attention: "Quel patch regarde quel autre patch"
  → C'est là que le modèle apprend les PATTERNS
  → Modifier ça change beaucoup le comportement

MLP: "Traitement générique des features"
  → Moins spécifique à la tâche
  → Moins besoin de le modifier

MAIS on PEUT aussi mettre LoRA sur MLP si on veut!
C'est un choix de design. Nous on a choisi qkv + proj.
```

---

## 4. C'est quoi Projection Layer (PROJ)?

### Contexte

```
Dans Self-Attention, on a plusieurs "têtes" (heads):

Multi-Head Attention:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Head 1: Regarde les relations de forme                        │
│  Head 2: Regarde les relations de texture                      │
│  Head 3: Regarde les relations de position                     │
│  Head 4: Regarde les relations de couleur                      │
│  Head 5: ...                                                    │
│  Head 6: ...                                                    │
│                                                                 │
│  6 têtes qui regardent des choses DIFFÉRENTES!                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Le problème

```
Après Multi-Head Attention, on a 6 résultats différents.
Comment les COMBINER en un seul résultat cohérent?

Head 1 output: [64 dims]
Head 2 output: [64 dims]    →  Concaténer → [384 dims]
Head 3 output: [64 dims]          ↓
Head 4 output: [64 dims]         PROJ
Head 5 output: [64 dims]          ↓
Head 6 output: [64 dims]    →  Sortie [384 dims]
```

### Rôle de PROJ

```
PROJ = Linear(384, 384)

Fonction:
1. MÉLANGER les informations des différentes têtes
2. APPRENDRE quelle combinaison est la meilleure
3. Produire une sortie COHÉRENTE

Sans PROJ: Les têtes ne communiquent pas entre elles
Avec PROJ: Le modèle apprend à combiner intelligemment
```

---

## 5. Explication de "Sortie = W×entrée + Δ"

### L'équation complète

```
SANS LoRA:
┌─────────────────────────────────────────┐
│  sortie = W × entrée                    │
│                                         │
│  W = matrice de poids [384×384]        │
│  entrée = vecteur [384]                │
│  sortie = vecteur [384]                │
└─────────────────────────────────────────┘

AVEC LoRA:
┌─────────────────────────────────────────┐
│  sortie = W × entrée + Δ               │
│                                         │
│  Où Δ = (B × A) × entrée               │
│                                         │
│  = W × entrée + B × A × entrée         │
│  = (W + B×A) × entrée                  │
└─────────────────────────────────────────┘
```

### Visualisation

```
                    SANS LoRA

entrée [384] ──→ [ × W ] ──→ sortie [384]


                    AVEC LoRA

                ┌──→ [ × W ] ──────────────┐
entrée [384] ──┤                           ├──→ [+] ──→ sortie [384]
                └──→ [ × A ] ──→ [ × B ] ──┘
                         ↓           ↓
                      [16 dims]   [384 dims]

                    Δ = B × A × entrée
```

### Exemple numérique simplifié

```
entrée = [1, 2, 3]  (simplifié à 3 dims)

W = poids originaux (gelés)
A = LoRA down (compression)
B = LoRA up (expansion)

Calcul:
1. W × entrée = [10, 20, 30]           (sortie originale)
2. A × entrée = [5]                     (compressé à 1 dim)
3. B × [5] = [1, 2, 1]                  (Δ, la modification)
4. sortie finale = [10+1, 20+2, 30+1] = [11, 22, 31]

LoRA ajoute une PETITE modification Δ à la sortie originale!
```

---

## 6. Pourquoi LoRA sur Attention et pas MLP?

### Réponse courte

```
On PEUT mettre LoRA partout! C'est un CHOIX de design.

Notre choix: qkv + proj (dans l'attention)

Pourquoi?
1. L'attention est PLUS IMPORTANTE pour adapter le modèle
2. Moins de paramètres = plus rapide
3. C'est ce qui marche bien dans la littérature
```

### Comparaison

```
┌─────────────────────────────────────────────────────────────────┐
│                    OÙ METTRE LoRA?                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Option 1: Seulement qkv + proj (notre choix)                  │
│  ├─ Paramètres: ~442K                                          │
│  ├─ Rapide à entraîner                                         │
│  └─ Suffisant pour la plupart des tâches                       │
│                                                                 │
│  Option 2: qkv + proj + MLP (fc1, fc2)                         │
│  ├─ Paramètres: ~1.2M                                          │
│  ├─ Plus lent                                                   │
│  └─ Potentiellement meilleur pour tâches complexes             │
│                                                                 │
│  Option 3: Partout (toutes les couches)                        │
│  ├─ Paramètres: ~2M+                                           │
│  ├─ Très lent                                                   │
│  └─ Risque de sur-apprentissage                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Plus d'explications sur PROJ

### Analogie avec une équipe

```
Imagine une équipe de 6 experts qui analysent un visage:

Expert 1 (Head 1): "Je vois la forme du nez"
Expert 2 (Head 2): "Je vois la texture de la peau"
Expert 3 (Head 3): "Je vois la position des yeux"
Expert 4 (Head 4): "Je vois les ombres"
Expert 5 (Head 5): "Je vois les contours"
Expert 6 (Head 6): "Je vois la symétrie"

PROJ = Le chef d'équipe qui COMBINE tous les avis:
"D'accord, en combinant tout ça, voici la description finale du visage"

PROJ apprend COMMENT combiner ces informations de manière optimale.
```

---

## 8. C'est quoi Q, K, V? Comment ça marche?

### L'intuition

```
Q = Query (Question)     "Qu'est-ce que je cherche?"
K = Key (Clé)           "Qu'est-ce que j'ai à offrir?"
V = Value (Valeur)      "Quelle information je donne?"

Analogie: Recherche dans une bibliothèque

Query (Q):  "Je cherche des livres sur les chats"
Key (K):    Chaque livre a des mots-clés (titre, sujet)
Value (V):  Le contenu du livre

1. Comparer Q avec tous les K → Scores de similarité
2. Les livres avec K similaires à Q ont des scores élevés
3. Récupérer les V des livres avec les meilleurs scores
```

### Pour les images

```
Chaque patch devient Q, K, et V:

┌─────────────────────────────────────────────────────────────────┐
│  Patch "œil gauche":                                           │
│                                                                 │
│  Q (Query): "Je suis un œil, qui d'autre est similaire?"       │
│  K (Key):   "Je suis un œil gauche à cette position"           │
│  V (Value): "Voici mes caractéristiques: [0.2, 0.8, ...]"      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Le calcul d'attention

```
Entrée X [256 patches × 384 dims]
           │
           ▼
     ┌─────────────┐
     │    QKV      │  Une seule matrice qui produit Q, K, V
     │ [384→1152]  │  1152 = 384 × 3 (pour Q, K, V)
     └─────────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
    Q     K     V
 [384]  [384]  [384]
     │     │
     ▼     ▼
   ┌─────────┐
   │ Q × K^T │  Calculer les scores d'attention
   │ /√384   │  (qui regarde qui?)
   └─────────┘
        │
        ▼
   ┌─────────┐
   │ Softmax │  Normaliser les scores (somme = 1)
   └─────────┘
        │
        ▼
   Attention × V  →  Sortie (mélange pondéré des V)
```

### Exemple concret

```
Supposons 3 patches simplifiés:

Patch 1 (œil):     Q1, K1, V1
Patch 2 (nez):     Q2, K2, V2
Patch 3 (bouche):  Q3, K3, V3

Calcul pour Patch 1:
┌─────────────────────────────────────────────────────────────────┐
│  Score(1→1) = Q1 · K1 = 0.9  (très similaire à lui-même)       │
│  Score(1→2) = Q1 · K2 = 0.3  (un peu lié au nez)               │
│  Score(1→3) = Q1 · K3 = 0.1  (peu lié à la bouche)             │
│                                                                 │
│  Après Softmax: [0.7, 0.2, 0.1]                                │
│                                                                 │
│  Nouvelle représentation de Patch 1:                           │
│  = 0.7 × V1 + 0.2 × V2 + 0.1 × V3                             │
│                                                                 │
│  = Patch 1 enrichi par les infos des autres patches!           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Bases de données d'entraînement et de test

### Notre configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRAÎNEMENT                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset: CASIA-WebFace                                         │
│  ├─ 494,149 images                                             │
│  ├─ 10,572 identités (personnes différentes)                   │
│  ├─ ~47 images par personne en moyenne                         │
│  └─ Chemin: data/casia-webface/                                │
│                                                                 │
│  Utilisation: Apprendre à distinguer les visages               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TEST (Benchmarks)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LFW (Labeled Faces in the Wild)                            │
│     ├─ 6,000 paires de visages                                 │
│     ├─ Type: Général                                            │
│     └─ Question: "Ces 2 visages sont la même personne?"        │
│                                                                 │
│  2. CFP-FP (Celebrities Frontal-Profile)                       │
│     ├─ Visages frontaux vs profils                             │
│     └─ Type: Variation de POSE                                 │
│                                                                 │
│  3. AgeDB-30 (Age Database)                                    │
│     ├─ Même personne à différents âges                         │
│     └─ Type: Variation d'ÂGE                                   │
│                                                                 │
│  4. CALFW (Cross-Age LFW)                                      │
│     └─ Type: ÂGE                                               │
│                                                                 │
│  5. CPLFW (Cross-Pose LFW)                                     │
│     └─ Type: POSE                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pourquoi des datasets différents?

```
ENTRAÎNEMENT ≠ TEST (très important!)

Si on teste sur les mêmes données qu'on entraîne:
  → Le modèle peut "mémoriser" au lieu d'apprendre
  → Pas de garantie qu'il généralise

En utilisant des datasets DIFFÉRENTS pour le test:
  → On vérifie que le modèle a vraiment APPRIS
  → Les personnes dans LFW ne sont PAS dans CASIA-WebFace
```

---

## 10. Puis-je publier mon modèle sur HuggingFace?

### OUI, absolument!

```
┌─────────────────────────────────────────────────────────────────┐
│                    PUBLICATION HUGGINGFACE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tu PEUX publier:                                               │
│  ✓ Les poids LoRA/DA-LoRA (petits fichiers)                   │
│  ✓ Le code du modèle                                           │
│  ✓ Les résultats et benchmarks                                 │
│  ✓ Un demo/espace interactif                                   │
│                                                                 │
│  ATTENTION - Ne PAS publier:                                   │
│  ✗ Les poids complets de DINO/CLIP (appartiennent à Meta/OpenAI)│
│  ✗ Le dataset CASIA-WebFace (licence restrictive)             │
│                                                                 │
│  MAIS - C'est OK car:                                          │
│  → Les utilisateurs téléchargent DINO depuis Meta              │
│  → Tu publies SEULEMENT tes poids LoRA (ce que tu as entraîné) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comment publier

```python
# Installation
pip install huggingface_hub

# Connexion
huggingface-cli login

# Créer un repo et publier
from huggingface_hub import HfApi
api = HfApi()

# Upload ton modèle
api.upload_folder(
    folder_path="checkpoints/",
    repo_id="ton-username/AdaptFace-DALoRA",
    repo_type="model"
)
```

---

## 11. C'est quoi HuggingFace exactement?

### Description

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUGGINGFACE 🤗                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  C'est quoi?                                                    │
│  → Le "GitHub" pour les modèles de Machine Learning            │
│  → Plateforme de partage de modèles, datasets, et démos        │
│                                                                 │
│  Services principaux:                                           │
│                                                                 │
│  1. HuggingFace Hub (hub.huggingface.co)                       │
│     ├─ Télécharger des modèles pré-entraînés                   │
│     ├─ Publier tes propres modèles                             │
│     └─ Partager des datasets                                    │
│                                                                 │
│  2. Transformers (bibliothèque Python)                         │
│     ├─ Code pour utiliser les modèles                          │
│     └─ pip install transformers                                │
│                                                                 │
│  3. Spaces (démos interactives)                                │
│     └─ Créer une interface web pour ton modèle                 │
│                                                                 │
│  4. Datasets (bibliothèque)                                    │
│     └─ Accéder facilement aux datasets publics                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comment l'utiliser

```python
# Exemple: Charger un modèle depuis HuggingFace
from transformers import AutoModel

# Télécharge automatiquement le modèle
model = AutoModel.from_pretrained("facebook/dinov2-small")

# Exemple: Publier ton modèle
model.push_to_hub("ton-username/mon-modele")
```

### Fiabilité

```
HuggingFace est:
✓ Utilisé par Google, Meta, Microsoft, OpenAI
✓ Standard de l'industrie pour le ML
✓ Open source et gratuit
✓ Très bien documenté

Conseils:
1. Utilise les modèles "officiels" (vérifiés)
2. Lis les licences avant d'utiliser
3. Vérifie les métriques et benchmarks publiés
```

---

## 12. Prompt pour visualiser les Transformers

### Prompt pour DALL-E / Midjourney / Stable Diffusion

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT RECOMMANDÉ                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EN ANGLAIS (meilleurs résultats):                             │
│                                                                 │
│  "Technical diagram of Vision Transformer (ViT) architecture,  │
│   showing image patches flowing through self-attention blocks, │
│   with Query, Key, Value vectors clearly labeled,              │
│   clean minimalist scientific illustration style,              │
│   white background, professional technical drawing,            │
│   educational diagram, neural network visualization"           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prompts spécifiques

```
Pour Self-Attention:
"Diagram showing self-attention mechanism,
 multiple patches connected with weighted arrows,
 attention scores visualized as connection strengths,
 clean technical illustration, white background"

Pour LoRA:
"Technical diagram of LoRA low-rank adaptation,
 showing frozen weights W plus small matrices A and B,
 matrix factorization visualization,
 clean minimalist scientific diagram"

Pour l'architecture complète:
"Vision Transformer architecture diagram,
 input image split into patches,
 patches encoded and processed through transformer blocks,
 classification head at the end,
 technical blueprint style, labeled components"
```

### MEILLEURE OPTION: Utiliser des outils dédiés

```
Pour des diagrammes VRAIMENT clairs, utilise plutôt:

1. draw.io (gratuit)
   → https://draw.io
   → Tu dessines toi-même, contrôle total

2. Excalidraw (gratuit)
   → https://excalidraw.com
   → Style "dessiné à la main", joli

3. TikZ/LaTeX (pour papers)
   → Diagrammes vectoriels de qualité publication

4. Lucidchart
   → Diagrammes professionnels

5. Mermaid (dans Markdown)
   → Diagrammes en code texte
```

### Exemple de diagramme en texte (que tu peux copier)

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER (ViT)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image 224×224                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │ Patch Embed │  Découper en 256 patches de 14×14             │
│  │ + Position  │  Ajouter info de position                     │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  ╔═══════════════════════════════════════════════╗             │
│  ║           TRANSFORMER BLOCK (×12)             ║             │
│  ╠═══════════════════════════════════════════════╣             │
│  ║                                               ║             │
│  ║  ┌─────────────────────────────────────────┐ ║             │
│  ║  │         SELF-ATTENTION                  │ ║             │
│  ║  │  ┌─────┐                                │ ║             │
│  ║  │  │ QKV │ → Q, K, V                      │ ║             │
│  ║  │  └─────┘                                │ ║             │
│  ║  │      ↓                                  │ ║             │
│  ║  │  Attention = softmax(Q·K^T/√d) × V     │ ║             │
│  ║  │      ↓                                  │ ║             │
│  ║  │  ┌──────┐                               │ ║             │
│  ║  │  │ PROJ │ → Combiner les têtes          │ ║             │
│  ║  │  └──────┘                               │ ║             │
│  ║  └─────────────────────────────────────────┘ ║             │
│  ║                    ↓                         ║             │
│  ║  ┌─────────────────────────────────────────┐ ║             │
│  ║  │              MLP                        │ ║             │
│  ║  │  Linear → GELU → Linear                 │ ║             │
│  ║  └─────────────────────────────────────────┘ ║             │
│  ║                                               ║             │
│  ╚═══════════════════════════════════════════════╝             │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │   [CLS]     │  Token de classification                      │
│  │   Token     │  → Représentation globale de l'image          │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  Embedding [384 dims] → Projection → [512 dims]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Résumé des Points Clés

```
┌─────────────────────────────────────────────────────────────────┐
│                    CE QU'IL FAUT RETENIR                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Google Colab gratuit est PLUS LENT que ton PC!             │
│                                                                 │
│  2. Self-Attention = les patches se "regardent" entre eux      │
│                                                                 │
│  3. MLP = traitement individuel de chaque patch                │
│                                                                 │
│  4. PROJ = combiner les résultats des différentes "têtes"      │
│                                                                 │
│  5. LoRA ajoute Δ = B×A×entrée à la sortie originale          │
│                                                                 │
│  6. On peut mettre LoRA partout, c'est un choix                │
│                                                                 │
│  7. Q=Question, K=Clé, V=Valeur pour l'attention               │
│                                                                 │
│  8. Train sur CASIA-WebFace, Test sur LFW/CFP/AgeDB            │
│                                                                 │
│  9. Tu PEUX publier tes poids LoRA sur HuggingFace             │
│                                                                 │
│  10. HuggingFace = plateforme de partage de modèles ML         │
│                                                                 │
│  11. Pour visualiser: draw.io ou Excalidraw > AI générative    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document créé le 2025-12-22*
