"""
[ROLE]
Ce fichier génère les visualisations de synthèse reliant la marche dynamique au COP statique.

[RESPONSIBILITY]
- Créer une figure de pont conceptuel (Bridge to COP).
- Générer une visualisation comparative de signatures PD vs CO.
- Produire un schéma logique du pipeline de traitement.

[INPUTS]
- Résultats de classification et d'importance des phases précédentes.

[OUTPUTS]
- output/figures/cop_bridge/*.png
- output/figures/concept_map/*.png

[ASSUMPTIONS]
- Représentation graphique schématique (matplotlib).

[RISKS]
- La simplification peut masquer la complexité physique réelle du transfert de charge.

[DEPENDENCIES]
- matplotlib
- pandas
- numpy
- project.config
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.config import OUTPUT_DIR

_BRIDGE_DIR = OUTPUT_DIR / "figures" / "cop_bridge"
_CONCEPT_DIR = OUTPUT_DIR / "figures" / "concept_map"


def create_bridge_to_cop_figure():
    """
    @brief Génère une figure montrant les correspondances entre dynamique et statique.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    # Boîtes Dynamiques
    boxes_dyn = [
        (1, 6, "Asymétrie L/R\n(Charge total_L vs total_R)"),
        (1, 4, "Variabilité temporelle\n(Écart-type swing/stance)"),
        (1, 2, "Régularité des pas\n(Autocorrélation/Intervalle)"),
    ]

    # Boîtes Statiques (Cibles Balance)
    boxes_stat = [
        (7, 6, "Répartition de charge\n(Asymétrie posturale)"),
        (7, 4, "Vitesse du COP\n(Stabilité médio-latérale)"),
        (7, 2, "Surface d'oscillation\n(Sway area / Enveloppe)"),
    ]

    def draw_box(x, y, text, color):
        ax.add_patch(
            plt.Rectangle(
                (x - 1.2, y - 0.7),
                2.4,
                1.4,
                fill=True,
                color=color,
                alpha=0.2,
                ec="black",
                lw=1.5,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")

    for x, y, txt in boxes_dyn:
        draw_box(x, y, txt, "skyblue")
    for x, y, txt in boxes_stat:
        draw_box(x, y, txt, "salmon")

    # Flèches de correspondance
    ax.annotate(
        "",
        xy=(5.8, 6),
        xytext=(2.2, 6),
        arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
    )
    ax.annotate(
        "",
        xy=(5.8, 4),
        xytext=(2.2, 4),
        arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
    )
    ax.annotate(
        "",
        xy=(5.8, 2),
        xytext=(2.2, 2),
        arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
    )

    ax.text(
        4,
        7,
        "PONT CONCEPTUEL : DYNAMIQUE → STATIQUE",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        1,
        7.2,
        "MARCHE DYNAMIQUE\n(Signaux PhysioNet)",
        ha="center",
        color="blue",
        fontweight="bold",
    )
    ax.text(
        8.5,
        7.2,
        "BALANCE STATIQUE\n(Objectif Final COP)",
        ha="center",
        color="red",
        fontweight="bold",
    )

    # Texte central
    ax.text(
        4,
        4.2,
        "Transfert de signature\nphysiopathologique",
        ha="center",
        fontsize=9,
        fontstyle="italic",
    )

    ax.set_axis_off()
    _BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(_BRIDGE_DIR / "bridge_to_cop.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_concept_map():
    """
    @brief Génère le schéma logique du pipeline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    steps = [
        (1, 3, "Entrée\nSignal brut\n(Charge L/R)"),
        (3.5, 3, "Traitement\nSegmentation\npar pas"),
        (6, 3, "Extraction\nFeatures XAI\n(Asymétrie/CV)"),
        (8.5, 3, "Sortie\nIndicateur de\nRisque PD/CO"),
    ]

    for i, (x, y, txt) in enumerate(steps):
        ax.add_patch(plt.Circle((x, y), 0.8, color="lightgray", ec="black", alpha=0.5))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 0.8, 3),
                xytext=(x + 0.8, 3),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_title(
        "Architecture Logique : Du signal au Biomarqueur",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_axis_off()
    _CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(_CONCEPT_DIR / "logic_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Génération de la figure Bridge to COP...")
    create_bridge_to_cop_figure()
    print("Génération du schéma conceptuel...")
    create_concept_map()
    print(f"Terminé ! Figures sauvegardées dans {OUTPUT_DIR}/figures/")


if __name__ == "__main__":
    main()
