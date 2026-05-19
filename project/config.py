"""
[ROLE]
Ce fichier contient les configurations globales et les constantes du projet.

[RESPONSIBILITY]
- Définir les constantes de session et d'aléatoire.
- Centraliser les chemins de sortie par défaut.

[INPUTS]
- Aucun.

[OUTPUTS]
- Constantes utilisables par les autres modules.

[ASSUMPTIONS]
- Aucun.

[RISKS]
- Aucun.

[DEPENDENCIES]
- Aucun.
"""

from __future__ import annotations

from pathlib import Path

# Constantes globales
RANDOM_STATE = 42
SESSION = "01"

# Chemins
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"

# Sous-dossiers pour les figures
XAI_FIG_DIR = FIG_DIR / "xai"
VAL_FIG_DIR = FIG_DIR / "validation"
SEG_FIG_DIR = FIG_DIR / "segmentation"
GAIT_FIG_DIR = FIG_DIR / "gait_profiles"
ASYM_FIG_DIR = FIG_DIR / "asymmetry_heatmaps"
BRIDGE_FIG_DIR = FIG_DIR / "concept_map"
