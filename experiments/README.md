# Experiments Tracking

Ce dossier contient tous les entraînements et expériences du projet AdaptFace.

## Structure

```
experiments/
├── README.md                    # Ce fichier
├── experiments_index.md         # Index de toutes les expériences
│
├── exp_001_baseline_dinov2/     # Première expérience
│   ├── config.yaml              # Configuration utilisée
│   ├── results.md               # Résultats et analyse
│   ├── training_log.txt         # Log d'entraînement (copie)
│   └── plots/                   # Graphiques (optionnel)
│
├── exp_002_baseline_clip/       # Deuxième expérience
│   └── ...
│
└── exp_XXX_description/         # Futures expériences
```

## Convention de nommage

Format: `exp_XXX_description`
- `XXX`: Numéro séquentiel (001, 002, ...)
- `description`: Courte description (baseline_dinov2, dalora_3domains, etc.)

## Comment ajouter une expérience

1. Créer le dossier `exp_XXX_description/`
2. Copier le template de `config.yaml`
3. Après l'entraînement, remplir `results.md`
4. Mettre à jour `experiments_index.md`

## Expériences terminées

| # | Nom | Date | LFW Acc | Statut |
|---|-----|------|---------|--------|
| 001 | baseline_dinov2 | 2024-12-20 | 90.45% | Complete |

