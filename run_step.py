"""
[ROLE]
Point d'entrée du pipeline step-level — comparaison de modèles sur GaitPDB.

[RESPONSIBILITY]
- Lancer la comparaison de modèles au niveau du pas (RF, LogReg, KMeans, FCM, CNN 1D, HybridCNN).
- Évaluer via StratifiedGroupKFold (split par sujet, sans fuite de données).
- Exporter les résultats CSV et les graphiques dans output/etude_du_pas/.

[DEPENDENCIES]
- gaitpdb.model_step_comparison, gaitpdb.step_improvements
"""

from __future__ import annotations

from gaitpdb.step_improvements import run_improvements
from gaitpdb.model_step_comparison import run_step_comparison


def main():
    # Comparaison baseline (RF, LogReg, KMeans, FCM, CNN 1D, HybridCNN)
    hybrid_preds = run_step_comparison(session="1")
    # Améliorations : filtrage, features enrichies + capteurs, seuil optimal, stacking
    run_improvements(session="1", hybrid_preds_df=hybrid_preds)


if __name__ == "__main__":
    main()
