"""
[ROLE]
Validation finale consolidée avec reporting visuel optimisé (Publication-Ready).

[RESPONSIBILITY]
- Exécuter K-Fold et LOSO.
- Générer des courbes ROC et Precision-Recall avec AUC.
- Visualiser la généralisation inter-études (LOSO) avec des barres d'erreur.

[OUTPUTS]
- output/figures/validation/*.png
- output/*.csv

[DEPENDENCIES]
- matplotlib, seaborn, sklearn, project.viz_utils
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION, VAL_FIG_DIR
from project.features import STEP_FEATURES, build_feature_matrix
from project.viz_utils import FIG_STD, FIG_WIDE, clean_label, save_fig, setup_style

setup_style()

PARSIMONIOUS = [
    "std_asym",
    "mean_asym",
    "mean_abs_diff",
    "diff_auc",
    "cv_interval_L",
    "n_steps",
]
FINAL_FEATURES = list(set(PARSIMONIOUS + STEP_FEATURES))


def plot_performance_summary(kf_res: pd.DataFrame):
    """@brief Barplot des métriques moyennes."""
    df_melt = kf_res.melt(var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=FIG_STD)
    sns.barplot(
        data=df_melt,
        x="Metric",
        y="Score",
        ax=ax,
        errorbar="sd",
        palette="viridis",
        hue="Metric",
    )
    ax.set_title("Performance Moyenne (5-Fold Cross-Validation)")
    ax.set_ylim(0, 1.05)
    save_fig(fig, VAL_FIG_DIR, "val_kfold_summary")


def plot_loso_performance(loso_res: pd.DataFrame):
    """@brief Performance par étude (LOSO) optimisée."""
    fig, ax = plt.subplots(figsize=FIG_STD)
    df_melt = loso_res.melt(
        id_vars="held_out_study", var_name="Metric", value_name="Score"
    )
    sns.barplot(
        data=df_melt,
        x="held_out_study",
        y="Score",
        hue="Metric",
        ax=ax,
        palette="muted",
    )
    ax.set_title("Robustesse Inter-Études (LOSO)")
    ax.set_xlabel("Étude exclue du train")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_fig(fig, VAL_FIG_DIR, "val_loso_robustness")


def plot_model_diagnostics(clf, X_test, y_test):
    """@brief Courbes ROC, PR et Matrice de Confusion."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ROC
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=axes[0], color="tab:orange")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Courbe ROC")

    # Precision-Recall
    PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, ax=axes[1], color="tab:blue"
    )
    axes[1].set_title("Courbe Precision-Recall")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, ax=axes[2], cmap="Blues", display_labels=["CO", "PD"]
    )
    axes[2].set_title("Matrice de Confusion")
    axes[2].grid(False)

    save_fig(fig, VAL_FIG_DIR, "val_diagnostic_curves")


def run_validation():
    print(f"--- Optimisation Validation - Session: {SESSION} ---")
    df = build_feature_matrix(session=SESSION)
    clf_df = df.dropna(subset=FINAL_FEATURES).copy()
    clf_df["label"] = (clf_df["group"] == "PD").astype(int)
    X, y = clf_df[FINAL_FEATURES].values, clf_df["label"].values

    # 1. K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kf_metrics = []
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])
        y_prob = clf.predict_proba(X[te])[:, 1]
        kf_metrics.append(
            {
                "Accuracy": accuracy_score(y[te], clf.predict(X[te])),
                "ROC-AUC": roc_auc_score(y[te], y_prob),
                "F1-Score": f1_score(y[te], clf.predict(X[te]), average="weighted"),
            }
        )
    plot_performance_summary(pd.DataFrame(kf_metrics))

    # 2. LOSO
    loso_results = []
    for study in sorted(clf_df["study"].unique()):
        tr_df, te_df = (
            clf_df[clf_df["study"] != study],
            clf_df[clf_df["study"] == study],
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(tr_df[FINAL_FEATURES].values, (tr_df["group"] == "PD").astype(int))
        y_te = (te_df["group"] == "PD").astype(int)
        y_prob = clf.predict_proba(te_df[FINAL_FEATURES].values)[:, 1]
        loso_results.append(
            {
                "held_out_study": study,
                "Accuracy": accuracy_score(
                    y_te, clf.predict(te_df[FINAL_FEATURES].values)
                ),
                "ROC-AUC": roc_auc_score(y_te, y_prob),
            }
        )
    plot_loso_performance(pd.DataFrame(loso_results))

    # 3. Diagnostics
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    plot_model_diagnostics(clf, X_te, y_te)

    print(f"Validation Terminée. Figures dans {VAL_FIG_DIR}")


def main():
    run_validation()


if __name__ == "__main__":
    main()
