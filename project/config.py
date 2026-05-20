##
# @file config.py
# @brief Ce fichier contient les configurations globales et les constantes du projet.
# @details Centralise les chemins de sortie, les états d'aléatoire et les paramètres de session.
#

from __future__ import annotations

from pathlib import Path

## @brief État d'aléatoire pour la reproductibilité.
RANDOM_STATE = 42

## @brief Session de marche retenue (01 = marche normale).
SESSION = "01"

# Chemins
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"

# Sous-dossiers pour les figures
XAI_FIG_DIR = FIG_DIR / "xai"
VAL_FIG_DIR = FIG_DIR / "validation"
MODEL_FIG_DIR = FIG_DIR / "model_comparison"
CLUSTERING_FIG_DIR = FIG_DIR / "clustering"
FUZZY_FIG_DIR = FIG_DIR / "fuzzy_clustering"
SEG_FIG_DIR = FIG_DIR / "segmentation"
GAIT_FIG_DIR = FIG_DIR / "gait_profiles"
ASYM_FIG_DIR = FIG_DIR / "asymmetry_heatmaps"
BRIDGE_FIG_DIR = FIG_DIR / "concept_map"
SHAP_FIG_DIR = FIG_DIR / "shap"
