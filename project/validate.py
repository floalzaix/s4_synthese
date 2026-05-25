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

##
# @file validate.py
# @brief Validation finale consolidée avec reporting visuel optimisé.
# @details Exécute K-Fold et LOSO, génère les courbes de performance (ROC, PR).
#

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION, VAL_FIG_DIR
from project.features import FEATURE_COLS, STEP_FEATURES, build_feature_matrix
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
# Ordre déterministe garanti : PARSIMONIOUS d'abord, puis STEP_FEATURES.
# set() ne préserve pas l'ordre → les labels XAI pouvaient pointer vers les mauvaises features.
FINAL_FEATURES = PARSIMONIOUS + [f for f in STEP_FEATURES if f not in PARSIMONIOUS]

# Pool complet (33 features) utilisé comme entrée de la validation principale.
# Aucune pré-sélection n'a lieu avant la CV : SelectFromModel choisit les features
# sur chaque fold de train indépendamment.
# FINAL_FEATURES est conservé uniquement pour la compatibilité des modules exploratoires
# (clustering, XAI) qui n'émettent pas de claims de performance.
ALL_CANDIDATE_FEATURES: list[str] = FEATURE_COLS


def plot_performance_summary(kf_res: pd.DataFrame):
    """
    @brief Barplot des métriques moyennes.
    @param kf_res DataFrame des résultats K-Fold.
    """
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
    """
    @brief Performance par étude (LOSO) optimisée.
    @param loso_res DataFrame des résultats LOSO.
    """
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
    """
    @brief Courbes ROC, PR et Matrice de Confusion.
    @param clf Classifieur entraîné.
    @param X_test Matrice de test.
    @param y_test Labels de test.
    """
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


def build_intra_cv_pipeline() -> Pipeline:
    """
    @brief Construit un Pipeline sklearn avec sélection de features encapsulée.
    @details SelectFromModel est entraîné UNIQUEMENT sur le fold de train.
    Le fold de test ne participe jamais à la sélection des features.
    Le seuil 'mean' conserve les features dont l'importance dépasse la moyenne.
    @return Pipeline sklearn prêt à être fitté sur un fold.
    """
    return Pipeline([
        (
            "selector",
            SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
                threshold="mean",
            ),
        ),
        (
            "clf",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
    ])


def plot_feature_stability(sel_df: pd.DataFrame, all_features: list[str]):
    """
    @brief Barplot horizontal de la fréquence de sélection de chaque feature sur les folds.
    @param sel_df DataFrame fold × feature (valeurs 0/1).
    @param all_features Liste ordonnée de toutes les features candidates.
    """
    feature_cols = [c for c in all_features if c in sel_df.columns]
    freq = sel_df[feature_cols].mean().sort_values(ascending=False)

    colors = [
        "#2ca02c" if v >= 0.8 else "#ff7f0e" if v >= 0.6 else "#d62728"
        for v in freq.values
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(freq) * 0.35)))
    ax.barh(
        [clean_label(f) for f in freq.index],
        freq.values,
        color=colors,
        edgecolor="white",
    )
    ax.axvline(0.6, color="black", linestyle="--", alpha=0.7, label="Seuil 60 %")
    ax.axvline(0.8, color="navy", linestyle=":", alpha=0.7, label="Seuil 80 %")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Fréquence de sélection (sur les folds CV)")
    ax.set_title("Stabilité de Sélection Intra-CV (vert >=80%, orange >=60%, rouge <60%)")
    ax.legend(loc="lower right")
    save_fig(fig, VAL_FIG_DIR, "val_feature_selection_stability")


def _write_cv_report(
    kf_df: pd.DataFrame,
    df_stab: pd.DataFrame,
    stable_features: list[str],
    n_candidates: int,
) -> None:
    """
    @brief Écrit le rapport texte de validation intra-CV dans output/.
    """
    lines = [
        "=== VALIDATION PRINCIPALE : SÉLECTION INTRA-CV ===\n",
        f"Pool initial : {n_candidates} features (FEATURE_COLS complet, sans pré-sélection)",
        "Méthode : Pipeline(SelectFromModel(RF100, threshold='mean') → RF200)",
        "Garantie : SelectFromModel est fitté sur le fold de TRAIN uniquement.\n",
        "1. MÉTRIQUES K-FOLD (5-Fold Stratifié) :",
        f"   ROC-AUC  : {kf_df['ROC-AUC'].mean():.3f} ± {kf_df['ROC-AUC'].std():.3f}",
        f"   Accuracy : {kf_df['Accuracy'].mean():.3f} ± {kf_df['Accuracy'].std():.3f}",
        f"   F1-Score : {kf_df['F1-Score'].mean():.3f} ± {kf_df['F1-Score'].std():.3f}",
        "\n2. FEATURES STABLES (sélectionnées dans ≥60 % des folds) :",
    ]
    for feat in stable_features:
        row = df_stab[df_stab["feature"] == feat]
        if not row.empty:
            lines.append(f"   - {feat} : {row['selection_frequency'].values[0]:.0%}")
    lines += [
        "\n3. NOTE MÉTHODOLOGIQUE :",
        "   La liste FINAL_FEATURES (19 features hard-codées) n'est plus utilisée",
        "   pour les métriques de performance. Elle subsiste pour les modules",
        "   exploratoires (clustering, XAI) dont les conclusions sont interprétatives,",
        "   pas prédictives.",
    ]
    with open(OUTPUT_DIR / "cv_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_validation(df: pd.DataFrame | None = None):
    """
    @brief Exécute la pipeline de validation complète avec sélection intra-CV.
    @details K-Fold et LOSO utilisent un Pipeline(SelectFromModel → RF) pour garantir
    que la sélection de features est apprise sur le fold de train uniquement.
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    print(f"--- Validation Intra-CV avec sélection de features - Session: {SESSION} ---")
    if df is None:
        df = build_feature_matrix(session=SESSION)

    # Départ depuis le pool complet — aucune pré-sélection globale.
    clf_df = df.dropna(subset=ALL_CANDIDATE_FEATURES).copy()
    clf_df["label"] = (clf_df["group"] == "PD").astype(int)
    X = clf_df[ALL_CANDIDATE_FEATURES].values
    y = clf_df["label"].values

    print(
        f"  Pool candidat : {len(ALL_CANDIDATE_FEATURES)} features | "
        f"Sujets valides : {len(clf_df)} "
        f"(PD={clf_df['label'].sum()}, CO={(~clf_df['label'].astype(bool)).sum()})"
    )

    # 1. K-Fold avec traçabilité de la sélection par fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kf_metrics = []
    selection_per_fold = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        pipe = build_intra_cv_pipeline()
        # SelectFromModel s'entraîne sur X[tr] uniquement — X[te] n'est jamais vu.
        pipe.fit(X[tr], y[tr])

        support = pipe.named_steps["selector"].get_support()
        fold_record = {
            "fold": fold_idx + 1,
            "n_selected": int(support.sum()),
        }
        fold_record.update(
            {f: int(s) for f, s in zip(ALL_CANDIDATE_FEATURES, support)}
        )
        selection_per_fold.append(fold_record)

        y_pred = pipe.predict(X[te])
        y_prob = pipe.predict_proba(X[te])[:, 1]
        kf_metrics.append(
            {
                "Accuracy": accuracy_score(y[te], y_pred),
                "ROC-AUC": roc_auc_score(y[te], y_prob),
                "F1-Score": f1_score(y[te], y_pred, average="weighted"),
            }
        )

    kf_df = pd.DataFrame(kf_metrics)
    kf_df.to_csv(OUTPUT_DIR / "cv_validation_kfold.csv", index=False)
    plot_performance_summary(kf_df)

    # Stabilité de sélection : fréquence de sélection de chaque feature sur les 5 folds.
    sel_df = pd.DataFrame(selection_per_fold)
    sel_df.to_csv(OUTPUT_DIR / "cv_feature_selection.csv", index=False)

    feat_cols = [c for c in ALL_CANDIDATE_FEATURES if c in sel_df.columns]
    freq = sel_df[feat_cols].mean().sort_values(ascending=False)
    df_stab = freq.reset_index()
    df_stab.columns = ["feature", "selection_frequency"]
    df_stab.to_csv(OUTPUT_DIR / "cv_feature_stability.csv", index=False)

    stable_features = df_stab[df_stab["selection_frequency"] >= 0.6]["feature"].tolist()
    print(f"  Features stables (>= 60% folds) : {stable_features}")

    plot_feature_stability(sel_df, ALL_CANDIDATE_FEATURES)

    # 2. LOSO — même pipeline, sélection apprise sur chaque sous-population d'entraînement.
    loso_results = []
    for study in sorted(clf_df["study"].unique()):
        tr_mask = clf_df["study"] != study
        te_mask = clf_df["study"] == study

        X_tr_l = clf_df.loc[tr_mask, ALL_CANDIDATE_FEATURES].values
        y_tr_l = clf_df.loc[tr_mask, "label"].values
        X_te_l = clf_df.loc[te_mask, ALL_CANDIDATE_FEATURES].values
        y_te_l = clf_df.loc[te_mask, "label"].values

        pipe = build_intra_cv_pipeline()
        pipe.fit(X_tr_l, y_tr_l)
        y_prob = pipe.predict_proba(X_te_l)[:, 1]
        loso_results.append(
            {
                "held_out_study": study,
                "Accuracy": accuracy_score(y_te_l, pipe.predict(X_te_l)),
                "ROC-AUC": roc_auc_score(y_te_l, y_prob),
            }
        )
    plot_loso_performance(pd.DataFrame(loso_results))

    # 3. Diagnostics visuels (split unique, illustration uniquement)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pipe = build_intra_cv_pipeline()
    pipe.fit(X_tr, y_tr)
    plot_model_diagnostics(pipe, X_te, y_te)

    _write_cv_report(kf_df, df_stab, stable_features, len(ALL_CANDIDATE_FEATURES))

    print(f"Validation terminée. Figures : {VAL_FIG_DIR}")
    print(f"Sorties CSV   : {OUTPUT_DIR}/cv_*.csv")


def load_stable_features(min_freq: float = 0.6) -> list[str]:
    """
    @brief Retourne les features sélectionnées dans au moins `min_freq` des folds CV.
    @details Lit output/cv_feature_stability.csv produit par run_validation().
    Utilisé par les modules exploratoires (XAI, clustering) pour s'aligner sur
    l'espace discriminant identifié sans fuite d'information.
    @param min_freq Seuil de fréquence de sélection (défaut 0.6 = 60% des folds).
    @return list[str] Features stables, dans l'ordre décroissant de fréquence.
    """
    csv_path = OUTPUT_DIR / "cv_feature_stability.csv"
    if not csv_path.exists():
        warnings.warn(
            "cv_feature_stability.csv introuvable. "
            "Exécutez run_validation() d'abord. Repli sur PARSIMONIOUS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return PARSIMONIOUS.copy()
    df_stab = pd.read_csv(csv_path)
    stable = df_stab[df_stab["selection_frequency"] >= min_freq]["feature"].tolist()
    if not stable:
        warnings.warn(
            f"Aucune feature avec selection_frequency >= {min_freq}. "
            "Retour à la liste complète.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df_stab["feature"].tolist()
    return stable


def main():
    """
    @brief Point d'entrée principal pour la validation.
    """
    run_validation()


if __name__ == "__main__":
    main()
