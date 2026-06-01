"""
[ROLE]
Point d'entrée principal du projet gaitpdb - Reporting Visuel Complet.

[RESPONSIBILITY]
- Charger la matrice de features une seule fois (évite 7 re-calculs).
- Exécuter la validation robuste (K-Fold, LOSO).
- Benchmarker plusieurs algorithmes de ML.
- Analyser la réduction de l'espace de features (Phase 4).
- Réaliser une analyse exploratoire de clustering (PCA, t-SNE, KMeans).
- Générer l'atlas visuel complet (XAI, Segmentation, Gait Profiles).

[DEPENDENCIES]
- gaitpdb.*
"""

from __future__ import annotations

import gaitpdb.feature_reduction
import gaitpdb.clustering
import gaitpdb.model_comparison
import gaitpdb.qc.step_qc
import gaitpdb.validate
import gaitpdb.qc.visual_check
import gaitpdb.viz.advanced
import gaitpdb.viz.conceptual
import gaitpdb.xai.importance
from gaitpdb.config import SESSION
from gaitpdb.features import build_feature_matrix


def main():
    print("====================================================")
    print("   GAITPDB PIPELINE - ATLAS VISUEL COMPLET")
    print("====================================================\n")

    # Chargement unique — toutes les étapes supervisées/non-supervisées
    # utilisent le même DataFrame pour garantir la reproductibilité.
    print("[0/10] Chargement de la matrice de features (calcul unique)...")
    df = build_feature_matrix(session=SESSION)
    print(f"OK. {len(df)} sujets chargés (session {SESSION}).\n")

    # 1. Validation Finale (K-Fold, LOSO, Métriques)
    print("[1/10] Exécution de la Validation Finale...")
    gaitpdb.validate.run_validation(df=df)
    print("OK.\n")

    # 2. Comparaison Multi-Algorithmes
    print("[2/10] Comparaison des algorithmes (Benchmarking)...")
    gaitpdb.model_comparison.run_model_comparison(df=df)
    print("OK.\n")

    # 3. Clustering & Projections
    print("[3/10] Analyse de Clustering et Projections (PCA/t-SNE)...")
    gaitpdb.clustering.run_patient_clustering(df=df)
    print("OK.\n")

    # 4. Fuzzy Clustering
    print("[4/10] Analyse Fuzzy Clustering (Appartenance Graduelle)...")
    gaitpdb.clustering.run_final_fuzzy(df=df)
    print("OK.\n")

    # 5. Réduction de Features (Phase 4 — trade-off performance/interprétabilité)
    print("[5/10] Réduction de l'espace des features (Phase 4)...")
    gaitpdb.feature_reduction.run_phase4(df=df)
    print("OK.\n")

    # 6. XAI (Importance, Distributions, Corrélations)
    print("[6/10] Génération du rapport XAI...")
    gaitpdb.xai.importance.run_xai_analysis(df=df)
    print("OK.\n")

    # 7. Contrôle Qualité Segmentation (Detailed Plots)
    print("[7/10] Génération du contrôle qualité segmentation...")
    gaitpdb.qc.visual_check.main()
    print("OK.\n")

    # 8. Visualisations Avancées (Profiles, Heatmaps)
    print("[8/10] Génération des profils et heatmaps d'asymétrie...")
    gaitpdb.viz.advanced.main(df=df)
    print("OK.\n")

    # 9. Schémas Conceptuels
    print("[9/10] Génération des schémas conceptuels...")
    gaitpdb.viz.conceptual.main()
    print("OK.\n")

    # 10. QC visuel des pas pd_mismatch et high_asym (step-level)
    print("[10/10] QC visuel des pas anomaliques (pd_mismatch / high_asym)...")
    gaitpdb.qc.step_qc.main()
    print("OK.\n")

    print("==========================================================")
    print("   TERMINÉ : Toutes les figures sont dans output/figures/")
    print("==========================================================")


if __name__ == "__main__":
    main()
