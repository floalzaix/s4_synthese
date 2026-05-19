"""
[ROLE]
Point d'entrée principal du projet gaitpdb - Reporting Visuel Complet.

[RESPONSIBILITY]
- Exécuter la validation robuste.
- Générer l'atlas visuel complet (XAI, Validation, Segmentation, Advanced Viz).
- Centraliser les appels vers les modules project.*.

[DEPENDENCIES]
- project.validate
- project.xai
- project.visual_check
- project.viz_advanced
- project.viz_bridge
"""

from __future__ import annotations

import project.validate
import project.visual_check
import project.viz_advanced
import project.viz_bridge
import project.xai


def main():
    print("====================================================")
    print("   GAITPDB PIPELINE - ATLAS VISUEL COMPLET")
    print("====================================================\n")

    # 1. Validation Finale (K-Fold, LOSO, Métriques)
    print("[1/5] Exécution de la Validation Finale...")
    project.validate.main()
    print("OK.\n")

    # 2. XAI (Importance, Distributions, Corrélations)
    print("[2/5] Génération du rapport XAI...")
    project.xai.run_xai_analysis()
    print("OK.\n")

    # 3. Contrôle Qualité Segmentation (Detailed Plots)
    print("[3/5] Génération du contrôle qualité segmentation...")
    project.visual_check.main()
    print("OK.\n")

    # 4. Visualisations Avancées (Profiles, Heatmaps)
    print("[4/5] Génération des profils et heatmaps d'asymétrie...")
    project.viz_advanced.main()
    print("OK.\n")

    # 5. Schémas Conceptuels
    print("[5/5] Génération des schémas conceptuels...")
    project.viz_bridge.main()
    print("OK.\n")

    print("====================================================")
    print("   TERMINÉ : Toutes les figures sont dans output/figures/")
    print("====================================================")


if __name__ == "__main__":
    main()
