"""
[ROLE]
Ce fichier contient les fonctions de visualisation XAI (Explainable AI) pour le rapport final.

[RESPONSIBILITY]
- Générer un barplot des importances des caractéristiques.
- Générer des boxplots de comparaison PD vs CO pour les caractéristiques clés.
- Produire un tableau de synthèse interprétable.

[INPUTS]
- Matrice de caractéristiques complète (Phase 4).

[OUTPUTS]
- output/figures/xai_basic/*.png
- output/xai_summary.csv

[ASSUMPTIONS]
- Le modèle RandomForest (Phase 4) est utilisé pour l'importance.

[RISKS]
- Les importances peuvent varier légèrement selon le split (Random Forest).

[DEPENDENCIES]
- matplotlib
- seaborn
- pandas
- sklearn
- project.config
- project.features
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION
from project.features import STEP_FEATURES, build_feature_matrix

_FIG_DIR = OUTPUT_DIR / "figures" / "xai_basic"

# Features d'intérêt pour les boxplots (mélange parcimonieuses + par pas)
KEY_FEATURES = [
    "std_asym",
    "mean_asym",
    "mean_abs_diff",
    "diff_auc",
    "asym_stance",
    "asym_swing",
    "cv_swing_L",
    "n_steps",
]


def generate_importance_plot(df: pd.DataFrame, features: list[str]):
    """
    @brief Génère un barplot des importances MDI du RandomForest.
    """
    X = df[features].values
    y = (df["group"] == "PD").astype(int).values

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X, y)

    imp_df = pd.DataFrame(
        {"feature": features, "importance": clf.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=imp_df.head(10),
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 des caractéristiques discriminantes (RF Importance)")
    plt.xlabel("Importance (MDI)")
    plt.ylabel("Caractéristique")
    plt.tight_layout()

    _FIG_DIR.mkdir(exist_ok=True, parents=True)
    plt.savefig(_FIG_DIR / "feature_importance.png")
    plt.close()
    return imp_df


def generate_key_boxplots(df: pd.DataFrame):
    """
    @brief Génère des boxplots PD vs CO pour les caractéristiques clés.
    """
    # On filtre les colonnes présentes
    available_keys = [f for f in KEY_FEATURES if f in df.columns]

    n_cols = 2
    n_rows = (len(available_keys) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(available_keys):
        sns.boxplot(
            data=df,
            x="group",
            y=feat,
            ax=axes[i],
            palette={"PD": "salmon", "CO": "skyblue"},
            hue="group",
            legend=False,
        )
        axes[i].set_title(f"Distribution de {feat}")
        axes[i].set_xlabel("Groupe")
        axes[i].set_ylabel("Valeur")

    # Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "key_features_comparison.png")
    plt.close()


def main():
    print(f"Génération des visualisations XAI - Session: {SESSION}")
    df = build_feature_matrix(session=SESSION)

    # Liste complète des features évaluées en phase 4
    PARSIMONIOUS_FEATURES = [
        "std_asym",
        "mean_asym",
        "mean_abs_diff",
        "diff_auc",
        "cv_interval_L",
        "n_steps",
    ]
    ALL_EVAL_FEATURES = list(set(PARSIMONIOUS_FEATURES + STEP_FEATURES))

    clf_df = df.dropna(subset=ALL_EVAL_FEATURES).copy()

    # 1. Importance
    imp_df = generate_importance_plot(clf_df, ALL_EVAL_FEATURES)
    print("Barplot des importances généré.")

    # 2. Boxplots
    generate_key_boxplots(clf_df)
    print("Boxplots de comparaison générés.")

    # 3. Synthèse
    # Calcul simple de la direction de l'effet (moyenne PD / moyenne CO)
    summary = []
    for feat in imp_df["feature"]:
        m_pd = clf_df[clf_df["group"] == "PD"][feat].mean()
        m_co = clf_df[clf_df["group"] == "CO"][feat].mean()
        direction = "Augmenté (PD > CO)" if m_pd > m_co else "Diminué (PD < CO)"

        # Commentaires biomécaniques simplifiés
        comment = ""
        if "asym" in feat or "diff" in feat:
            comment = (
                "Reflète la perte de symétrie bilatérale caractéristique du Parkinson."
            )
        elif "cv_" in feat or "std" in feat:
            comment = (
                "Indique une instabilité ou irrégularité accrue du cycle de marche."
            )
        elif "n_steps" in feat or "cadence" in feat:
            comment = (
                "Lié à la vitesse de marche et à la stratégie globale de déplacement."
            )
        elif "stance" in feat:
            comment = "Traduit le temps passé en appui, souvent allongé chez les PD (prudence)."
        elif "swing" in feat:
            comment = (
                "Reflète la phase dynamique, souvent plus variable chez les patients."
            )

        summary.append(
            {
                "feature": feat,
                "importance": round(
                    imp_df[imp_df["feature"] == feat]["importance"].values[0], 4
                ),
                "direction": direction,
                "biomechanical_interest": comment,
            }
        )

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUTPUT_DIR / "xai_summary.csv", index=False)
    print(f"Synthèse sauvegardée dans {OUTPUT_DIR / 'xai_summary.csv'}")
    print("\nTop 5 features et direction :")
    print(
        df_summary.head(5)[["feature", "importance", "direction"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
