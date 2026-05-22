"""
[ROLE]
Comparaison exhaustive de multiples algorithmes de ML classés par familles.

[RESPONSIBILITY]
- Benchmarker >12 modèles (Arbres, Linéaires, Boosting, Baselines).
- Gérer les imports conditionnels pour XGBoost, LightGBM, CatBoost.
- Générer 6 PNG distincts (3 Barplots, 3 Boxplots) pour Accuracy, F1 et ROC-AUC.

[OUTPUTS]
- output/model_comparison_cv.csv
- output/figures/model_comparison/*.png

[DEPENDENCIES]
- scikit-learn
- xgboost, lightgbm, catboost (optionnels)
- project.config, project.features, project.validate, project.viz_utils
"""

##
# @file model_comparison.py
# @brief Comparaison exhaustive de multiples algorithmes de ML.
# @details Benchmarque différents modèles (RF, LogReg, SVC, boosters) et génère des rapports de performance.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from project.config import MODEL_FIG_DIR, OUTPUT_DIR, RANDOM_STATE
from project.features import build_feature_matrix
from project.validate import FINAL_FEATURES
from project.viz_utils import save_fig, setup_style


def get_external_boosters():
    """
    @brief Gère les imports conditionnels pour les boosters externes.
    @return dict Dictionnaire contenant les modèles XGBoost/LightGBM/CatBoost si installés.
    """
    boosters = {}
    try:
        from xgboost import XGBClassifier

        boosters["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss"
        )
    except ImportError:
        pass

    return boosters


setup_style()


def get_all_models() -> dict:
    """
    @brief Construit le dictionnaire de tous les modèles par famille.
    @return dict Dictionnaire {nom: modèle}.
    """
    # 1. Famille Arbres
    trees = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "GradBoost": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    # 2. Famille Linéaire / Marges
    linear = {
        "LogReg": Pipeline(
            [
                ("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("s", StandardScaler()),
                ("c", LinearSVC(dual=False, random_state=RANDOM_STATE)),
            ]
        ),
        "SVC-RBF": Pipeline(
            [
                ("s", StandardScaler()),
                ("c", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
    }

    # 4. Boosting Externes
    external = get_external_boosters()

    # Fusion
    all_m = {}
    all_m.update(trees)
    all_m.update(linear)
    all_m.update(external)
    return all_m


def generate_plots(df_cv: pd.DataFrame):
    """
    @brief Génère les barplots et boxplots de performance.
    @param df_cv DataFrame contenant les résultats de cross-validation.
    """
    metrics = ["Accuracy", "F1-Score", "ROC-AUC", "Balanced_Acc"]

    # Tri par score moyen pour la lisibilité
    order = (
        df_cv.groupby("model")["Balanced_Acc"].mean().sort_values(ascending=False).index
    )

    for metric in metrics:
        # Barplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_cv,
            x="model",
            y=metric,
            order=order,
            ax=ax,
            errorbar="sd",
            palette="viridis",
            hue="model",
            legend=False,
        )
        ax.set_title(f"Performance Moyenne : {metric} (5-Fold CV par sujet)")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        save_fig(fig, MODEL_FIG_DIR, f"baseline_barplot_{metric.lower()}")

        # Boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df_cv,
            x="model",
            y=metric,
            order=order,
            ax=ax,
            palette="Set3",
            hue="model",
            legend=False,
        )
        ax.set_title(f"Distribution des Scores : {metric} (Stabilité folds)")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        save_fig(fig, MODEL_FIG_DIR, f"baseline_boxplot_{metric.lower()}")


def run_model_comparison(session: str = "01", df: pd.DataFrame | None = None):
    """
    @brief Point d'entrée principal pour la comparaison des modèles.
    @param session Identifiant de la session (défaut "01").
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    print(f"--- Phase 1: Tabular Baseline - Session: {session} ---")
    print("NOTE MÉTHODOLOGIQUE :")
    print("- Les sessions '10' (dual_task, série Ga) sont explicitement exclues.")
    print(
        "- Seules les marches standards (session '01') sont retenues pour éviter la fuite de protocoles."
    )
    print(
        "- StratifiedKFold garantit une séparation étanche par sujet (1 ligne = 1 sujet)."
    )

    if df is None:
        df = build_feature_matrix(session=session)
    # FINAL_FEATURES : 19 features parcimonieuses pré-définies (subset de FEATURE_COLS).
    # Ce benchmark évalue les modèles sur cet espace fixe, sans sélection intra-CV.
    # Il est complémentaire à run_validation() (qui fait SelectFromModel sur 33 features)
    # et produit des comparaisons multi-algorithmes sur un espace homogène et reproductible.
    clf_df = df.dropna(subset=FINAL_FEATURES).copy()

    print(
        f"Sujets conservés : {len(clf_df)} (Distribution PD/CO: {dict(clf_df['group'].value_counts())})"
    )

    X = clf_df[FINAL_FEATURES].values
    y = (clf_df["group"] == "PD").astype(int).values

    models = get_all_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in models.items():
        fold = 1
        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            y_pred = model.predict(X[te])

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[te])[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X[te])
            else:
                y_prob = y_pred

            results.append(
                {
                    "model": name,
                    "fold": fold,
                    "Accuracy": accuracy_score(y[te], y_pred),
                    "Balanced_Acc": balanced_accuracy_score(y[te], y_pred),
                    "F1-Score": f1_score(y[te], y_pred, average="weighted"),
                    "ROC-AUC": roc_auc_score(y[te], y_prob),
                }
            )
            fold += 1

    df_cv = pd.DataFrame(results)
    df_cv.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    generate_plots(df_cv)

    summary = (
        df_cv.groupby("model")[["Accuracy", "Balanced_Acc", "ROC-AUC"]]
        .mean()
        .sort_values("Balanced_Acc", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "phase1_baseline_summary.csv")
    print("\nPhase 1 Baseline (Moyenne CV par sujet) :")
    print(summary.to_string())


if __name__ == "__main__":
    run_model_comparison()
