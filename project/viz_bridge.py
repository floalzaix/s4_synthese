"""
[ROLE]
Schémas conceptuels harmonisés pour le rapport final.

[RESPONSIBILITY]
- Générer le pont conceptuel Dynamique -> Statique.
- Générer la map logique du pipeline.

[OUTPUTS]
- output/figures/concept_map/*.png
"""

##
# @file viz_bridge.py
# @brief Schémas conceptuels harmonisés.
# @details Génère des diagrammes expliquant le pipeline de traitement et le pont entre dynamique et statique.
#

from __future__ import annotations

import matplotlib.pyplot as plt

from project.config import BRIDGE_FIG_DIR
from project.viz_utils import PALETTE, save_fig, setup_style

setup_style()


def plot_bridge():
    """
    @brief Génère le schéma du pont conceptuel Dynamique -> Statique.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    boxes_dyn = [
        (1, 6, "Asymétrie L/R"),
        (1, 4, "Variabilité\nTemporelle"),
        (1, 2, "Régularité"),
    ]
    boxes_stat = [
        (7, 6, "Charge Posturale"),
        (7, 4, "Vitesse COP"),
        (7, 2, "Surface Sway"),
    ]

    def box(x, y, txt, col):
        ax.add_patch(
            plt.Rectangle(
                (x - 1.2, y - 0.7), 2.4, 1.4, color=col, alpha=0.15, ec="black", lw=1.5
            )
        )
        ax.text(x, y, txt, ha="center", va="center", weight="bold")

    for x, y, t in boxes_dyn:
        box(x, y, t, PALETTE["CO"])
    for x, y, t in boxes_stat:
        box(x, y, t, PALETTE["PD"])
    for y in [2, 4, 6]:
        ax.annotate(
            "",
            xy=(5.8, y),
            xytext=(2.2, y),
            arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
        )

    ax.text(
        5,
        7.5,
        "PONT CONCEPTUEL : DYNAMIQUE → STATIQUE",
        ha="center",
        size=14,
        weight="bold",
    )
    ax.set_axis_off()
    save_fig(fig, BRIDGE_FIG_DIR, "final_bridge_to_cop")


def plot_logic():
    """
    @brief Génère le schéma de l'architecture logique du pipeline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    steps = [
        (1, 3, "Signaux"),
        (3.5, 3, "Segmentation"),
        (6, 3, "Features"),
        (8.5, 3, "Score Risk"),
    ]
    for i, (x, y, t) in enumerate(steps):
        ax.add_patch(plt.Circle((x, y), 0.8, color="lightgray", ec="black", alpha=0.3))
        ax.text(x, y, t, ha="center", va="center", weight="bold")
        if i < 3:
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 0.8, 3),
                xytext=(x + 0.8, 3),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_title("Architecture Logique du Pipeline", weight="bold")
    ax.set_axis_off()
    save_fig(fig, BRIDGE_FIG_DIR, "final_logic_pipeline")


def main():
    """
    @brief Point d'entrée principal pour les schémas conceptuels.
    """
    plot_bridge()
    plot_logic()


if __name__ == "__main__":
    main()
