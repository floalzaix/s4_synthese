"""
[ROLE]
Analyse SHAP multi-fold pour l'interprétabilité du classifieur PD vs CO.

[RESPONSIBILITY]
- Reproduire les 5 folds CV de validate.py (mêmes paramètres, mêmes données).
- Calculer les SHAP values sur le fold de TEST uniquement (pas de fuite).
- Agréger par feature : moyenne/std inter-folds, rang, direction PD/CO.
- Comparer SHAP / Permutation / MDI.
- Générer figures et CSV dans output/.

[INPUTS]
- clf_df     : DataFrame préparé (dropna sur xai_features, même que xai.py).
- xai_features : liste des features stables (issues de cv_feature_stability.csv).

[OUTPUTS]
- output/shap_values.csv           : valeurs par sujet et par fold.
- output/shap_summary.csv          : agrégation inter-folds par feature.
- output/shap_importance_comparison.csv : SHAP vs perm vs MDI.
- output/figures/shap/*.png        : barplot, stabilité, heatmap groupe.

[ASSUMPTIONS]
- shap >= 0.40 installé (pip install shap).
- RandomForestClassifier avec n_estimators=200, random_state=RANDOM_STATE.
- 5-fold StratifiedKFold avec shuffle=True, random_state=RANDOM_STATE.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

try:
    import shap as _shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from gaitpdb.config import OUTPUT_DIR, RANDOM_STATE, SHAP_FIG_DIR
from gaitpdb.viz.utils import clean_label, save_fig, setup_style

setup_style()

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ──────────────────────────────────────────────────────────────────────────────

def _require_shap() -> None:
    """Lance une ImportError lisible si shap n'est pas installé."""
    if not HAS_SHAP:
        raise ImportError(
            "Le package 'shap' est requis. Installez-le avec : pip install shap"
        )


def _extract_shap_class1(shap_values) -> np.ndarray:
    """
    Normalise la sortie de TreeExplainer.shap_values() vers (n_samples, n_features).
    shap 0.51+ retourne un ndarray (n, f, 2) pour RF binaire.
    Versions antérieures retournent une liste de deux arrays.
    """
    if isinstance(shap_values, list):
        # ancienne API : [shap_class0, shap_class1]
        return np.asarray(shap_values[1])
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # nouvelle API : (n_samples, n_features, n_classes) — classe 1 = PD
        return sv[:, :, 1]
    return sv  # fallback : déjà (n_samples, n_features)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _plot_mean_barplot(df_summary: pd.DataFrame) -> None:
    """
    Barplot horizontal des mean_shap ordonnés par |mean_shap|.
    Barres rouges = pousse vers PD, bleues = pousse vers CO.
    Barres d'erreur = std inter-folds (stabilité).
    """
    df_plot = df_summary.sort_values("mean_abs_shap", ascending=True)
    labels = [clean_label(f) for f in df_plot["feature"]]
    means = df_plot["mean_shap"].values
    stds = df_plot["std_shap"].values
    colors = ["salmon" if v >= 0 else "skyblue" for v in means]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_plot) * 0.55)))
    ax.barh(labels, means, xerr=stds, color=colors, edgecolor="white", capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP moyen (+= vers PD, -= vers CO)  |  barre d'erreur = std inter-folds")
    ax.set_title("Contribution SHAP par Feature (agrege sur 5 folds)")
    save_fig(fig, SHAP_FIG_DIR, "shap_mean_barplot")


def _plot_stability(df_shap: pd.DataFrame, xai_features: list[str]) -> None:
    """
    Boxplot des moyennes SHAP par fold pour chaque feature.
    Illustre la variabilité inter-folds : une large dispersion indique
    une interprétation peu stable d'un fold à l'autre.
    """
    shap_cols = [f"shap_{f}" for f in xai_features]
    fold_means = (
        df_shap.groupby("fold")[shap_cols]
        .mean()
        .reset_index()
        .melt(id_vars="fold", var_name="col", value_name="mean_shap")
    )
    fold_means["label"] = fold_means["col"].str.replace("shap_", "", regex=False).apply(clean_label)

    order = (
        fold_means.groupby("label")["mean_shap"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(xai_features) * 0.65)))
    sns.boxplot(
        data=fold_means,
        x="mean_shap",
        y="label",
        order=order,
        ax=ax,
        hue="label",
        palette="muted",
        legend=False,
        width=0.45,
    )
    # Chaque point = 1 fold (n=5)
    sns.stripplot(
        data=fold_means,
        x="mean_shap",
        y="label",
        order=order,
        ax=ax,
        color="black",
        size=5,
        alpha=0.6,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP moyen par fold")
    ax.set_title(
        "Stabilite inter-folds des Valeurs SHAP\n"
        "(chaque point = moyenne sur le fold de test ; dispersion = instabilite)"
    )
    save_fig(fig, SHAP_FIG_DIR, "shap_stability_by_feature")


def _plot_group_heatmap(df_shap: pd.DataFrame, xai_features: list[str]) -> None:
    """
    Heatmap SHAP moyen par feature × groupe (PD / CO).
    Pour les sujets PD bien classés, on attend des SHAP positifs sur les features
    discriminantes ; pour les CO, des SHAP négatifs sur les mêmes features.
    """
    shap_cols = [f"shap_{f}" for f in xai_features]
    group_means = df_shap.groupby("group")[shap_cols].mean().T
    group_means.index = [
        clean_label(c.replace("shap_", "")) for c in group_means.index
    ]
    # Réordonner par |SHAP moyen| global
    group_means["abs_mean"] = group_means.abs().mean(axis=1)
    group_means = group_means.sort_values("abs_mean", ascending=False).drop(
        columns="abs_mean"
    )

    fig, ax = plt.subplots(figsize=(5, max(5, len(xai_features) * 0.55)))
    sns.heatmap(
        group_means,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "SHAP moyen"},
    )
    ax.set_title("SHAP Moyen par Feature et par Groupe\n(+= pousse vers PD)")
    ax.set_xlabel("Groupe clinique")
    save_fig(fig, SHAP_FIG_DIR, "shap_group_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    clf_df: pd.DataFrame,
    xai_features: list[str],
) -> pd.DataFrame:
    """
    @brief Analyse SHAP multi-fold (TreeExplainer) sur les features stables.
    @details Reproduit exactement les 5 folds de validate.py. Pour chaque fold :
      - RF refitté sur X_train avec les features stables,
      - SHAP calculé sur X_test uniquement (pas de fuite).
    Les valeurs sont agrégées inter-folds (moyenne/std) et comparées aux importances MDI/perm.
    @param clf_df DataFrame préparé avec dropna sur xai_features (même que xai.py).
    @param xai_features Liste des features stables (cv_feature_stability.csv, freq>=0.6).
    @return pd.DataFrame df_summary avec mean_shap, std_shap, mean_abs_shap, rank.
    """
    _require_shap()

    X = clf_df[xai_features].values
    y = (clf_df["group"] == "PD").astype(int).values
    subject_ids = clf_df["subject_id"].values
    groups = clf_df["group"].values

    print(
        f"  SHAP : {len(xai_features)} features | "
        f"{len(clf_df)} sujets (PD={y.sum()}, CO={(~y.astype(bool)).sum()})"
    )

    # Reproduction exacte des splits de validate.py (même graine, mêmes données)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_records: list[dict] = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        # RF identique à xai.py — pas de re-sélection de features ici
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])

        # SHAP calculé exclusivement sur le fold de TEST
        explainer = _shap_lib.TreeExplainer(clf)
        shap_raw = explainer.shap_values(X[te])
        shap_pd = _extract_shap_class1(shap_raw)  # (n_test, n_features)

        for i, test_idx in enumerate(te):
            record: dict = {
                "fold": fold_idx + 1,
                "subject_id": subject_ids[test_idx],
                "group": groups[test_idx],
                "y_true": int(y[test_idx]),
            }
            for j, feat in enumerate(xai_features):
                record[f"shap_{feat}"] = float(shap_pd[i, j])
            all_records.append(record)

    df_shap = pd.DataFrame(all_records)
    df_shap.to_csv(OUTPUT_DIR / "shap_values.csv", index=False)

    # ── Agrégation inter-folds ────────────────────────────────────────────────
    shap_cols = [f"shap_{f}" for f in xai_features]

    # std_shap = std des moyennes de fold (variabilité inter-folds, pas intra-sujet)
    fold_means = df_shap.groupby("fold")[shap_cols].mean()

    summary_rows = []
    for feat, col in zip(xai_features, shap_cols):
        summary_rows.append(
            {
                "feature": feat,
                "mean_shap": float(fold_means[col].mean()),
                "std_shap": float(fold_means[col].std()),
                "mean_abs_shap": float(df_shap[col].abs().mean()),
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        "mean_abs_shap", ascending=False
    )
    df_summary["rank"] = range(1, len(df_summary) + 1)
    df_summary.to_csv(OUTPUT_DIR / "shap_summary.csv", index=False)

    # ── Comparaison SHAP / MDI / Perm ────────────────────────────────────────
    fi_path = OUTPUT_DIR / "feature_importance_metrics.csv"
    if fi_path.exists():
        df_fi = pd.read_csv(fi_path)[
            ["feature", "importance_mdi", "importance_perm_mean"]
        ]
        df_cmp = df_summary[
            ["feature", "mean_shap", "std_shap", "mean_abs_shap", "rank"]
        ].merge(df_fi, on="feature", how="left")
        df_cmp.to_csv(OUTPUT_DIR / "shap_importance_comparison.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_mean_barplot(df_summary)
    _plot_stability(df_shap, xai_features)
    _plot_group_heatmap(df_shap, xai_features)

    print(f"  SHAP figures -> {SHAP_FIG_DIR}")
    print(f"  SHAP CSV     -> {OUTPUT_DIR}/shap_*.csv")
    return df_summary


if __name__ == "__main__":
    from gaitpdb.features import build_feature_matrix
    from gaitpdb.validate import load_stable_features

    _df = build_feature_matrix(session="01")
    _feats = load_stable_features(min_freq=0.6)
    _clf_df = _df.dropna(subset=_feats).copy()
    run_shap_analysis(_clf_df, _feats)
