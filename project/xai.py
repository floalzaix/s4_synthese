"""
[ROLE]
Ce fichier contient les fonctions de visualisation XAI (Explainable AI) optimisées.

[RESPONSIBILITY]
- Générer un dashboard d'importance (MDI et Permutation).
- Visualiser les distributions des caractéristiques discriminantes.
- Analyser les corrélations avec des labels propres.
- Produire un rapport de synthèse interprétable.

[OUTPUTS]
- output/figures/xai/*.png
- output/xai_summary.csv

[DEPENDENCIES]
- matplotlib, seaborn, pandas, sklearn, project.viz_utils
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION, XAI_FIG_DIR
from project.features import STEP_FEATURES, build_feature_matrix
from project.viz_utils import (
    FIG_LARGE,
    FIG_STD,
    FIG_WIDE,
    PALETTE,
    clean_label,
    save_fig,
    setup_style,
)

setup_style()


def plot_importance_dashboard(mdi_imp: pd.DataFrame, perm_imp: pd.DataFrame):
    """@brief Dashboard comparatif des importances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)

    # Clean features for plotting
    mdi_plot = mdi_imp.head(12).copy()
    mdi_plot["feature_clean"] = mdi_plot["feature"].apply(clean_label)
    perm_plot = perm_imp.head(12).copy()
    perm_plot["feature_clean"] = perm_plot["feature"].apply(clean_label)

    # MDI
    sns.barplot(
        data=mdi_plot,
        x="importance",
        y="feature_clean",
        ax=ax1,
        hue="feature_clean",
        palette="viridis",
        legend=False,
    )
    ax1.set_title("Importance Modèle (RF MDI)")
    ax1.set_xlabel("Importance Relative")
    ax1.set_ylabel("")

    # Permutation
    sns.barplot(
        data=perm_plot,
        x="importance_mean",
        y="feature_clean",
        ax=ax2,
        hue="feature_clean",
        palette="magma",
        legend=False,
    )
    ax2.errorbar(
        perm_plot["importance_mean"],
        np.arange(len(perm_plot)),
        xerr=perm_plot["importance_std"],
        fmt="none",
        c="black",
        capsize=3,
    )
    ax2.set_title("Importance par Permutation (Test Set)")
    ax2.set_xlabel("Chute de Performance (Accuracy)")
    ax2.set_ylabel("")

    save_fig(fig, XAI_FIG_DIR, "xai_importance_dashboard")


def plot_top_distributions(df: pd.DataFrame, top_features: list[str]):
    """@brief Violinplots des top caractéristiques."""
    n = len(top_features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        sns.violinplot(
            data=df,
            x="group",
            y=feat,
            hue="group",
            ax=axes[i],
            palette=PALETTE,
            split=True,
            inner="quart",
            legend=False,
        )
        axes[i].set_title(clean_label(feat))
        axes[i].set_ylabel("Valeur")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    save_fig(fig, XAI_FIG_DIR, "xai_top_distributions")


def plot_correlation_matrix(df: pd.DataFrame, features: list[str]):
    """@brief Heatmap de corrélation avec labels propres."""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True
    )
    ax.set_xticklabels([clean_label(l) for l in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([clean_label(l) for l in corr.index])
    ax.set_title("Matrice de Corrélation des Caractéristiques Clés")
    save_fig(fig, XAI_FIG_DIR, "xai_feature_correlation")


def run_xai_analysis():
    print(f"--- Optimisation XAI - Session: {SESSION} ---")
    df = build_feature_matrix(session=SESSION)

    # Selection Features
    PARSIMONIOUS = [
        "std_asym",
        "mean_asym",
        "mean_abs_diff",
        "diff_auc",
        "cv_interval_L",
        "n_steps",
    ]
    ALL_FEATURES = list(set(PARSIMONIOUS + STEP_FEATURES))
    clf_df = df.dropna(subset=ALL_FEATURES).copy()
    clf_df["label"] = (clf_df["group"] == "PD").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        clf_df[ALL_FEATURES],
        clf_df["label"],
        test_size=0.25,
        stratify=clf_df["label"],
        random_state=RANDOM_STATE,
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Calcul Importances
    mdi_imp = pd.DataFrame(
        {"feature": ALL_FEATURES, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
    )
    perm_imp = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Figures
    plot_importance_dashboard(mdi_imp, perm_imp)
    plot_top_distributions(clf_df, mdi_imp.head(9)["feature"].tolist())

    key_feats = ["std_asym", "asym_swing", "asym_stance", "mean_asym", "n_steps"]
    plot_correlation_matrix(clf_df, [f for f in key_feats if f in ALL_FEATURES])

    # CSV Summary
    summary = []
    for feat in mdi_imp["feature"]:
        m_pd = clf_df[clf_df["group"] == "PD"][feat].mean()
        m_co = clf_df[clf_df["group"] == "CO"][feat].mean()
        summary.append(
            {
                "feature": feat,
                "label": clean_label(feat),
                "importance_mdi": round(
                    mdi_imp.loc[mdi_imp["feature"] == feat, "importance"].values[0], 4
                ),
                "direction": "PD > CO" if m_pd > m_co else "CO > PD",
            }
        )
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "xai_summary.csv", index=False)
    print(f"XAI Terminé. Figures dans {XAI_FIG_DIR}")


if __name__ == "__main__":
    run_xai_analysis()
