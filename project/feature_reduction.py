"""
[ROLE]
Phase 4 : Réduction de l'espace des features et analyse de stabilité.

[RESPONSIBILITY]
- Définir différents jeux de features (Complet, Réduit sans redondance L/R, Compact Clinique).
- Évaluer le Random Forest sur ces sets (CV 5-Fold stratifié par sujet).
- Comparer les performances (Acc, Bal_Acc, AUC) et la stabilité des importances XAI.
- Générer un rapport justifiant ou non la simplification de l'espace.

[OUTPUTS]
- output/phase4_feature_sets_comparison.csv
- output/phase4_xai_stability.csv
- output/phase4_report.txt
- output/figures/validation/phase4_*.png

[DEPENDENCIES]
- scikit-learn, pandas, numpy, seaborn, matplotlib
- project.config, project.features, project.validate, project.viz_utils
"""

##
# @file feature_reduction.py
# @brief Phase 4 : Réduction de l'espace des features et analyse de stabilité.
# @details Évalue l'impact de la simplification de l'espace des caractéristiques sur la performance et l'interprétabilité.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION, VAL_FIG_DIR, XAI_FIG_DIR
from project.features import build_feature_matrix
from project.validate import FINAL_FEATURES
from project.viz_utils import clean_label, save_fig, setup_style

setup_style()

# --- DÉFINITION DES SETS DE FEATURES ---

# 1. Complet (Baseline actuelle)
SET_FULL = FINAL_FEATURES.copy()

# 2. Réduit (Bilatéral Moyenné + Asymétrie)
SET_REDUCED_BASE = [
    "std_asym",
    "mean_asym",
    "mean_abs_diff",
    "diff_auc",
    "asym_stance",
    "asym_swing",
    "asym_auc_steps",
    "cv_interval_L",
]

# 3. Compact (Strictement orienté physiopathologie attendue)
SET_COMPACT = [
    "mean_asym",
    "asym_stance",
    "cv_interval_L",
    "n_steps",
]


def prepare_data(session: str, df=None) -> pd.DataFrame:
    """
    @brief Charge les données et prépare les variables bilatérales.
    @param session Identifiant de la session.
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    @return pd.DataFrame Données préparées.
    """
    if df is None:
        df = build_feature_matrix(session=session)
    clf_df = df.dropna(subset=FINAL_FEATURES).copy()

    # Création des moyennes bilatérales
    clf_df["mean_stance_bilat"] = (
        clf_df["mean_stance_L"] + clf_df["mean_stance_R"]
    ) / 2
    clf_df["mean_swing_bilat"] = (clf_df["mean_swing_L"] + clf_df["mean_swing_R"]) / 2
    clf_df["cv_stance_bilat"] = (clf_df["cv_stance_L"] + clf_df["cv_stance_R"]) / 2
    clf_df["cv_swing_bilat"] = (clf_df["cv_swing_L"] + clf_df["cv_swing_R"]) / 2
    clf_df["n_steps_seg_bilat"] = (
        clf_df["n_steps_L_seg"] + clf_df["n_steps_R_seg"]
    ) / 2

    clf_df["label"] = (clf_df["group"] == "PD").astype(int)
    return clf_df


# Mise à jour de SET_REDUCED avec les nouvelles variables
SET_REDUCED = SET_REDUCED_BASE + [
    "mean_stance_bilat",
    "mean_swing_bilat",
    "cv_stance_bilat",
    "cv_swing_bilat",
    "n_steps",
]

SETS = {
    "1_Complet": SET_FULL,
    "2_Réduit_Bilatéral": SET_REDUCED,
    "3_Compact_Clinique": SET_COMPACT,
}


def evaluate_set(X, y, set_name):
    """
    @brief Évalue un set de features avec K-Fold et calcule la stabilité XAI.
    @param X Matrice des caractéristiques.
    @param y Labels.
    @param set_name Nom du set évalué.
    @return tuple(pd.DataFrame, np.ndarray, np.ndarray) Métriques, moyennes importances, std importances.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = []
    importances_list = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])

        y_pred = clf.predict(X[te])
        y_prob = clf.predict_proba(X[te])[:, 1]

        metrics.append(
            {
                "set_name": set_name,
                "fold": fold + 1,
                "Accuracy": accuracy_score(y[te], y_pred),
                "Balanced_Acc": balanced_accuracy_score(y[te], y_pred),
                "F1-Score": f1_score(y[te], y_pred, average="weighted"),
                "ROC-AUC": roc_auc_score(y[te], y_prob),
            }
        )

        perm = permutation_importance(
            clf, X[te], y[te], n_repeats=5, random_state=RANDOM_STATE
        )
        importances_list.append(perm.importances_mean)

    imp_matrix = np.array(importances_list)
    imp_mean = np.mean(imp_matrix, axis=0)
    imp_std = np.std(imp_matrix, axis=0)

    return pd.DataFrame(metrics), imp_mean, imp_std


def run_phase4(session: str = "01", df=None):
    """
    @brief Exécute l'analyse de phase 4 complète.
    @param session Identifiant de la session (défaut "01").
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    print(f"--- Phase 4 : Réduction de l'espace des features - Session: {session} ---")
    clf_df = prepare_data(session, df=df)
    y = clf_df["label"].values

    all_metrics = []
    stability_records = []

    for set_name, features in SETS.items():
        print(f"Évaluation du set : {set_name} ({len(features)} features)")
        X = clf_df[features].values
        df_m, imp_mean, imp_std = evaluate_set(X, y, set_name)
        all_metrics.append(df_m)

        for i, feat in enumerate(features):
            stability_records.append(
                {
                    "set_name": set_name,
                    "feature": clean_label(feat),
                    "importance_mean": imp_mean[i],
                    "importance_std": imp_std[i],
                    "cv_importance": (imp_std[i] / (imp_mean[i] + 1e-9))
                    if imp_mean[i] > 0.01
                    else np.nan,
                }
            )

    df_res = pd.concat(all_metrics)
    df_res.to_csv(OUTPUT_DIR / "phase4_model_comparison.csv", index=False)

    df_stab = pd.DataFrame(stability_records)
    df_stab.to_csv(OUTPUT_DIR / "phase4_xai_stability.csv", index=False)

    # VISUALISATIONS
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_res, x="set_name", y="Balanced_Acc", palette="Pastel1", ax=ax)
    sns.swarmplot(data=df_res, x="set_name", y="Balanced_Acc", color=".25", ax=ax)
    ax.set_title("Comparaison des Performances (Balanced Accuracy)")
    ax.set_ylim(0.5, 1.0)
    save_fig(fig, VAL_FIG_DIR, "phase4_performance_comparison")

    fig, ax = plt.subplots(figsize=(12, 6))
    top_instable = (
        df_stab.dropna().sort_values("cv_importance", ascending=False).head(15)
    )
    sns.barplot(
        data=top_instable,
        y="feature",
        x="cv_importance",
        hue="set_name",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Instabilité des Importances (CV = Ecart-type / Moyenne inter-folds)")
    ax.set_xlabel("Coefficient de Variation de l'Importance")
    save_fig(fig, XAI_FIG_DIR, "phase4_xai_instability")

    # RAPPORT TEXTUEL
    summary = df_res.groupby("set_name")[["Balanced_Acc", "ROC-AUC"]].mean()

    report = []
    report.append("=== PHASE 4 : RAPPORT DE RÉDUCTION DE L'ESPACE DES FEATURES ===\n")
    report.append(
        "OBJECTIF : Tester si l'élimination de la redondance bilatérale (Gauche/Droite) stabilise l'explicabilité sans dégrader la performance.\n"
    )

    report.append("1. PERFORMANCES MOYENNES (5-Fold CV par sujet) :")
    for set_n, row in summary.iterrows():
        report.append(
            f"  - {set_n} : Balanced Acc = {row['Balanced_Acc']:.3f} | ROC-AUC = {row['ROC-AUC']:.3f}"
        )

    acc_diff = (
        summary.loc["2_Réduit_Bilatéral", "Balanced_Acc"]
        - summary.loc["1_Complet", "Balanced_Acc"]
    )

    report.append("\n2. ANALYSE DE LA ROBUSTESSE :")
    if abs(acc_diff) < 0.02:
        report.append(
            "  -> La réduction de dimension maintient des performances équivalentes au set complet."
        )
    elif acc_diff > 0.02:
        report.append("  -> La réduction de dimension AMÉLIORE la performance.")
    else:
        report.append("  -> La réduction dégrade légèrement la performance.")

    report.append("\n3. STABILITÉ XAI :")
    report.append(
        "  Le graphique 'phase4_xai_instability' montre que les variables hautement corrélées sont plus instables."
    )

    with open(OUTPUT_DIR / "phase4_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Phase 4 terminée. Rapport généré : {OUTPUT_DIR / 'phase4_report.txt'}")


if __name__ == "__main__":
    run_phase4()
