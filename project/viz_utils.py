"""
[ROLE]
Utilitaire centralisé pour la cohérence graphique du projet.

[RESPONSIBILITY]
- Définir la palette de couleurs (PD vs CO).
- Configurer le style Seaborn/Matplotlib.
- Fournir des fonctions de sauvegarde standardisées.

[DEPENDENCIES]
- matplotlib
- seaborn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Palette de couleurs officielle
PALETTE = {"PD": "salmon", "CO": "skyblue"}
COLOR_PD = PALETTE["PD"]
COLOR_CO = PALETTE["CO"]

# Mapping pour des labels propres en publication
LABEL_MAP = {
    "std_asym": "Asymmetry (Std Dev)",
    "mean_asym": "Mean Asymmetry",
    "mean_abs_diff": "Mean Abs. Difference",
    "diff_auc": "AUC Difference (L-R)",
    "cv_interval_L": "CV Step Interval (L)",
    "n_steps": "Total Step Count",
    "asym_stance": "Stance Asymmetry",
    "asym_swing": "Swing Asymmetry",
    "cv_swing_L": "CV Swing Time (L)",
    "cadence_spm": "Cadence (steps/min)",
    "group": "Group",
    "study": "Study Site",
}

# Tailles standard
FIG_STD = (10, 6)
FIG_WIDE = (15, 8)
FIG_LARGE = (14, 10)


def clean_label(label: str) -> str:
    """@brief Retourne un label lisible à partir d'une clé technique."""
    return LABEL_MAP.get(label, str(label).replace("_", " ").title())


def setup_style():
    """@brief Configure le style global pour les graphiques."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
        }
    )


def save_fig(fig, path: Path, name: str):
    """@brief Sauvegarde une figure de manière propre."""
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / f"{name}.png"
    fig.tight_layout()
    fig.savefig(full_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return full_path


def get_palette():
    return PALETTE
