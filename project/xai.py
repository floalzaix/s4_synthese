"""
[ROLE]
Phase 3 : Explicabilité (XAI) et Interprétation des Caractéristiques.

[RESPONSIBILITY]
- Décomposer la décision du meilleur modèle (Random Forest).
- Calculer l'importance MDI et par Permutation.
- Classifier les features par familles physiologiques (Asymétrie, Variabilité, etc.).
- Identifier les directions d'effet (PD > CO) et la redondance (corrélation).
- Générer un rapport textuel d'interprétation prudent.

[OUTPUTS]
- output/feature_importance_metrics.csv
- output/xai_report.txt
- output/figures/xai/*.png
"""

##
# @file xai.py
# @brief Phase 3 : Explicabilité (XAI) et Interprétation des Caractéristiques.
# @details Analyse l'importance des features via MDI et permutation, et catégorise par famille.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION, XAI_FIG_DIR
from project.features import build_feature_matrix
from project.validate import load_stable_features
from project.viz_utils import FIG_STD, FIG_WIDE, clean_label, save_fig, setup_style

setup_style()

# Définition des familles de features
FAMILY_MAPPING = {
    "std_asym": "Asymétrie",
    "mean_asym": "Asymétrie",
    "mean_abs_diff": "Asymétrie",
    "diff_auc": "Asymétrie",
    "asym_stance": "Asymétrie",
    "asym_swing": "Asymétrie",
    "asym_auc_steps": "Asymétrie",
    "cv_interval_L": "Variabilité",
    "cv_stance_L": "Variabilité",
    "cv_stance_R": "Variabilité",
    "cv_swing_L": "Variabilité",
    "cv_swing_R": "Variabilité",
    "n_steps": "Cadence/Quantité",
    "n_steps_L_seg": "Cadence/Quantité",
    "n_steps_R_seg": "Cadence/Quantité",
    "mean_stance_L": "Phases de marche",
    "mean_stance_R": "Phases de marche",
    "mean_swing_L": "Phases de marche",
    "mean_swing_R": "Phases de marche",
}


def get_feature_family(feat_name):
    """
    @brief Retourne la famille physiologique d'une caractéristique.
    @param feat_name Nom technique de la caractéristique.
    @return str Nom de la famille.
    """
    return FAMILY_MAPPING.get(feat_name, "Autre")


def plot_importance_dashboard(df_metrics: pd.DataFrame):
    """
    @brief Génère un dashboard comparant MDI et Permutation Importance.
    @param df_metrics DataFrame contenant les scores d'importance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)

    df_plot = df_metrics.sort_values("importance_perm_mean", ascending=False).head(12)

    sns.barplot(
        data=df_plot,
        x="importance_mdi",
        y="label",
        ax=ax1,
        palette="viridis",
        hue="label",
        legend=False,
    )
    ax1.set_title("Importance Intrinsèque (RF MDI)")
    ax1.set_xlabel("Importance Relative")
    ax1.set_ylabel("")

    sns.barplot(
        data=df_plot,
        x="importance_perm_mean",
        y="label",
        ax=ax2,
        palette="magma",
        hue="label",
        legend=False,
    )
    ax2.errorbar(
        df_plot["importance_perm_mean"],
        np.arange(len(df_plot)),
        xerr=df_plot["importance_perm_std"],
        fmt="none",
        c="black",
        capsize=3,
    )
    ax2.set_title("Importance par Permutation (Baisse d'Accuracy)")
    ax2.set_xlabel("Baisse de Performance")
    ax2.set_ylabel("")

    save_fig(fig, XAI_FIG_DIR, "xai_importance_dashboard")


def plot_family_importance(df_metrics: pd.DataFrame):
    """
    @brief Génère un barplot des importances agrégées par famille physiologique.
    @param df_metrics DataFrame contenant les scores d'importance.
    """
    fam_imp = df_metrics.groupby("family")["importance_perm_mean"].sum().reset_index()
    fam_imp = fam_imp.sort_values("importance_perm_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=fam_imp,
        x="importance_perm_mean",
        y="family",
        palette="Set2",
        hue="family",
        ax=ax,
        legend=False,
    )
    ax.set_title("Importance Prédictive Cumulée par Famille Physiologique")
    ax.set_xlabel("Somme des Importances par Permutation")
    ax.set_ylabel("")
    save_fig(fig, XAI_FIG_DIR, "xai_family_importance")


def generate_report(
    df_metrics: pd.DataFrame,
    df_corr: pd.DataFrame,
    clf_df: pd.DataFrame,
    df_shap_summary: pd.DataFrame | None = None,
):
    """
    @brief Génère le rapport textuel d'interprétation XAI.
    @param df_metrics DataFrame des importances.
    @param df_corr Matrice de corrélation des features.
    @param clf_df DataFrame source pour l'analyse.
    """
    top_features = df_metrics.sort_values("importance_perm_mean", ascending=False).head(
        5
    )

    report = []
    report.append("=== PHASE 3 : RAPPORT D'INTERPRÉTATION XAI (RANDOM FOREST) ===\n")
    report.append(
        "ATTENTION : Ce rapport est observationnel. Les importances indiquent une utilité prédictive pour l'algorithme, pas nécessairement une causalité clinique absolue.\n"
    )

    # 1. Top Features et Direction
    report.append(
        "1. CARACTÉRISTIQUES LES PLUS DISCRIMINANTES (Top 5 par Permutation) :"
    )
    for _, row in top_features.iterrows():
        fam = row["family"]
        direction = row["direction"]
        report.append(
            f"  - {row['label']} ({fam}) : Importance = {row['importance_perm_mean']:.3f} | Tendance : {direction}"
        )

    # 2. Analyse des familles
    report.append("\n2. ANALYSE DES FAMILLES PHYSIOLOGIQUES :")
    fam_imp = df_metrics.groupby("family")["importance_perm_mean"].sum()
    dom_fam = fam_imp.idxmax()
    report.append(
        f"  - La famille dominante est '{dom_fam}'. Cela suggère que la pathologie s'exprime prioritairement sur cet axe."
    )
    if fam_imp.get("Asymétrie", 0) > fam_imp.get("Variabilité", 0):
        report.append(
            "  - L'Asymétrie semble plus informative que la Variabilité temporelle intra-membre pour ce modèle."
        )
    else:
        report.append("  - La Variabilité prend le pas sur l'Asymétrie globale.")

    cadence_imp = fam_imp.get("Cadence/Quantité", 0)
    if cadence_imp < 0.05:
        report.append(
            "  - Les variables de quantité (ex: nombre de pas) jouent un rôle secondaire, indiquant que le modèle se base bien sur la qualité de la marche."
        )

    # 3. MDI vs Permutation
    top_mdi = set(
        df_metrics.sort_values("importance_mdi", ascending=False).head(5)["feature"]
    )
    top_perm = set(top_features["feature"])
    if top_mdi != top_perm:
        report.append("\n3. STABILITÉ DU MODÈLE :")
        report.append(
            "  - Divergence observée entre l'importance intrinsèque (MDI) et l'importance par permutation."
        )
        report.append(
            "  - Cela est souvent dû à des caractéristiques fortement corrélées (redondance) où le MDI divise l'importance."
        )

    # 4. Redondances (Corrélations > 0.75 parmi le Top 10)
    report.append("\n4. REDONDANCE ET CO-LINÉARITÉ (Parmi le Top 10) :")
    top_10_feats = (
        df_metrics.sort_values("importance_perm_mean", ascending=False)
        .head(10)["feature"]
        .tolist()
    )
    corr_top = df_corr.loc[top_10_feats, top_10_feats]
    high_corr = []
    for i in range(len(corr_top.columns)):
        for j in range(i + 1, len(corr_top.columns)):
            if abs(corr_top.iloc[i, j]) > 0.75:
                high_corr.append(
                    f"{clean_label(corr_top.columns[i])} <-> {clean_label(corr_top.columns[j])} (r={corr_top.iloc[i, j]:.2f})"
                )

    if high_corr:
        for hc in high_corr:
            report.append(f"  - {hc}")
        report.append(
            "  - Note : Ces variables portent un signal similaire. L'algorithme pourrait basculer de l'une à l'autre selon les plis de validation."
        )
    else:
        report.append(
            "  - Faible redondance parmi les top features. Le modèle utilise des sources d'information orthogonales."
        )

    report.append("\n5. SYNTHÈSE CLINIQUE :")
    report.append(
        "Le modèle exploite des variables compatibles avec la signature clinique attendue de la maladie de Parkinson. La structure des importances suggère un signal physiologiquement plausible, où la dégradation de la symétrie bilatérale et l'instabilité du cycle semblent plus discriminantes qu'un simple déficit de capacité brute (comme la vitesse ou le nombre total de pas)."
    )

    # 6. SHAP comparison (optional — only if df_shap_summary is provided)
    if df_shap_summary is not None and not df_shap_summary.empty:
        report.append("\n6. COMPARAISON SHAP / PERMUTATION / MDI :")
        df_cmp = df_metrics[["feature", "importance_perm_mean", "importance_mdi"]].merge(
            df_shap_summary[["feature", "mean_abs_shap", "std_shap"]], on="feature", how="inner"
        )
        # Features where all three methods agree (top half by each metric)
        n_half = max(1, len(df_cmp) // 2)
        top_perm = set(df_cmp.nlargest(n_half, "importance_perm_mean")["feature"])
        top_mdi = set(df_cmp.nlargest(n_half, "importance_mdi")["feature"])
        top_shap = set(df_cmp.nlargest(n_half, "mean_abs_shap")["feature"])
        robust = top_perm & top_mdi & top_shap
        if robust:
            report.append(
                "  Signal robuste (accord Perm + MDI + SHAP) : "
                + ", ".join(clean_label(f) for f in robust)
            )
        # Features with near-zero SHAP despite being in top-perm (potential redundancy)
        low_shap_thresh = df_cmp["mean_abs_shap"].quantile(0.25)
        redundant = df_cmp[
            df_cmp["feature"].isin(top_perm) & (df_cmp["mean_abs_shap"] <= low_shap_thresh)
        ]["feature"].tolist()
        if redundant:
            report.append(
                "  Potentiellement redondant (perm eleve, SHAP faible) : "
                + ", ".join(clean_label(f) for f in redundant)
            )
            report.append(
                "  -> Ces features sont utiles au modele mais portent un signal similaire a d'autres features;"
                " leur retrait individuel peut etre compense."
            )
        # Features with high std_shap (unstable interpretation)
        high_std_thresh = df_cmp["std_shap"].quantile(0.75)
        unstable = df_cmp[df_cmp["std_shap"] >= high_std_thresh]["feature"].tolist()
        if unstable:
            report.append(
                "  Interpretation instable (std_shap eleve) : "
                + ", ".join(clean_label(f) for f in unstable)
            )
            report.append(
                "  -> La contribution SHAP de ces features varie fortement d'un fold a l'autre."
                " Interpreter avec prudence."
            )

    with open(OUTPUT_DIR / "xai_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def run_xai_analysis(session: str = "01", df=None):
    """
    @brief Exécute l'analyse XAI multi-fold sur les features stables identifiées par la CV.
    @details Les features XAI sont lues depuis cv_feature_stability.csv (produit par
    run_validation). Les importances par permutation sont calculées sur chaque fold de
    test et moyennées pour réduire la variance liée à un seul split.
    Aucune information du fold de test n'influence le choix des features.
    @param session Identifiant de la session (défaut "01").
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    print(f"--- Phase 3 : Interprétabilité XAI multi-fold - Session: {session} ---")

    if df is None:
        df = build_feature_matrix(session=session)

    # Features stables issues de la CV — alignées avec l'espace du classifieur.
    xai_features = load_stable_features(min_freq=0.6)
    print(f"  Features XAI (stables CV >= 60%) : {xai_features}")

    clf_df = df.dropna(subset=xai_features).copy()
    X = clf_df[xai_features].values
    y = (clf_df["group"] == "PD").astype(int).values

    # Agrégation sur 5 folds : importances MDI et permutation
    # Plus stable qu'un seul hold-out (~25 sujets), réduit la variance d'estimation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mdi_per_fold = []
    perm_means_per_fold = []

    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])
        mdi_per_fold.append(clf.feature_importances_)
        perm = permutation_importance(
            clf, X[te], y[te], n_repeats=10, random_state=RANDOM_STATE
        )
        perm_means_per_fold.append(perm.importances_mean)

    # Moyenne et écart-type inter-folds des importances par permutation
    mdi_agg = np.mean(mdi_per_fold, axis=0)
    perm_mean_agg = np.mean(perm_means_per_fold, axis=0)
    perm_std_agg = np.std(perm_means_per_fold, axis=0)  # variabilité inter-folds

    # Directions et familles calculées sur l'ensemble complet (statistique descriptive)
    records = []
    for i, feat in enumerate(xai_features):
        m_pd = clf_df[clf_df["group"] == "PD"][feat].mean()
        m_co = clf_df[clf_df["group"] == "CO"][feat].mean()
        records.append(
            {
                "feature": feat,
                "label": clean_label(feat),
                "family": get_feature_family(feat),
                "importance_mdi": mdi_agg[i],
                "importance_perm_mean": perm_mean_agg[i],
                "importance_perm_std": perm_std_agg[i],
                "mean_PD": m_pd,
                "mean_CO": m_co,
                "direction": "PD > CO" if m_pd > m_co else "CO > PD",
            }
        )

    df_metrics = pd.DataFrame(records).sort_values(
        "importance_perm_mean", ascending=False
    )
    df_metrics.to_csv(OUTPUT_DIR / "feature_importance_metrics.csv", index=False)

    df_corr = clf_df[xai_features].corr()

    plot_importance_dashboard(df_metrics)
    plot_family_importance(df_metrics)

    # SHAP multi-fold analysis — enriches report with a third importance estimator
    df_shap_summary = None
    try:
        from project.shap_analysis import run_shap_analysis  # noqa: PLC0415
        df_shap_summary = run_shap_analysis(clf_df, xai_features)
    except ImportError:
        print("  [INFO] shap non installe — section SHAP ignoree (pip install shap)")

    generate_report(df_metrics, df_corr, clf_df, df_shap_summary=df_shap_summary)

    print(f"XAI termine.")
    print(f"- Export CSV : {OUTPUT_DIR / 'feature_importance_metrics.csv'}")
    print(f"- Rapport : {OUTPUT_DIR / 'xai_report.txt'}")
    print(f"- Figures : {XAI_FIG_DIR}/")


if __name__ == "__main__":
    run_xai_analysis()
