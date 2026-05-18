"""
[ROLE]
Ce fichier orchestre la validation finale et consolidée du pipeline gaitpdb.

[RESPONSIBILITY]
- Définir le set de caractéristiques final (Parsimonieux + Par Pas).
- Exécuter la validation croisée (K-Fold 5-splits) pour mesurer la stabilité.
- Exécuter la validation inter-études (LOSO) pour mesurer la généralisation.
- Calculer les importances par permutation sur le modèle final.
- Sauvegarder les résultats consolidés dans le dossier output/.

[INPUTS]
- Matrice de caractéristiques complète (project.features.build_feature_matrix).

[OUTPUTS]
- output/final_validation_summary.csv
- output/final_loso_results.csv
- output/final_feature_importance.csv

[ASSUMPTIONS]
- Session 01 uniquement.
- RandomForestClassifier (200 estimateurs) comme modèle de référence.

[RISKS]
- Sensibilité à la variabilité inter-études (adressée par LOSO).

[DEPENDENCIES]
- pandas
- numpy
- sklearn
- project.config
- project.features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from project.config import OUTPUT_DIR, RANDOM_STATE, SESSION
from project.features import STEP_FEATURES, build_feature_matrix

# --- 1. Définition du set de caractéristiques final ---
PARSIMONIOUS_BASE = [
    "std_asym",
    "mean_asym",
    "mean_abs_diff",
    "diff_auc",
    "cv_interval_L",
    "n_steps",
]

# Set final combinant le global parsimonieux et les descripteurs fins par pas
FINAL_FEATURES = list(set(PARSIMONIOUS_BASE + STEP_FEATURES))


def run_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    """
    @brief Exécute une validation K-Fold et retourne le détail des métriques.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        metrics.append(
            {
                "accuracy": accuracy_score(y_te, y_pred),
                "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
                "roc_auc": roc_auc_score(y_te, y_prob),
            }
        )
    return pd.DataFrame(metrics)


def run_loso(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    @brief Exécute une validation Leave-One-Study-Out.
    """
    studies = sorted(df["study"].unique())
    results = []
    for study in studies:
        train_df = df[df["study"] != study].copy()
        test_df = df[df["study"] == study].copy()

        X_tr = train_df[features].values
        y_tr = (train_df["group"] == "PD").astype(int).values
        X_te = test_df[features].values
        y_te = (test_df["group"] == "PD").astype(int).values

        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        results.append(
            {
                "held_out_study": study,
                "accuracy": accuracy_score(y_te, y_pred),
                "roc_auc": roc_auc_score(y_te, y_prob),
            }
        )
    return pd.DataFrame(results)


def main():
    print(f"Validation Consolidée - Session: {SESSION}")
    df = build_feature_matrix(session=SESSION)

    # Nettoyage NaN sur le set final
    clf_df = df.dropna(subset=FINAL_FEATURES).copy()
    clf_df["label"] = (clf_df["group"] == "PD").astype(int)
    X = clf_df[FINAL_FEATURES].values
    y = clf_df["label"].values

    print(f"Nombre de sujets valides : {len(clf_df)}")
    print(f"Nombre de caractéristiques : {len(FINAL_FEATURES)}")

    # 1. K-Fold
    print("\n--- 1. Validation K-Fold (Meilleur Modèle) ---")
    kf_res = run_kfold(X, y)
    summary = kf_res.mean().to_frame(name="mean")
    summary["std"] = kf_res.std()
    summary.to_csv(OUTPUT_DIR / "final_validation_summary.csv")
    print(summary.to_string())

    # 2. LOSO
    print("\n--- 2. Validation LOSO (Généralisation inter-études) ---")
    loso_res = run_loso(clf_df, FINAL_FEATURES)
    loso_res.to_csv(OUTPUT_DIR / "final_loso_results.csv", index=False)
    print(loso_res.to_string(index=False))

    # 3. Importance
    print("\n--- 3. Importance des caractéristiques (Permutation) ---")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    perm = permutation_importance(
        clf, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp_df = pd.DataFrame(
        {
            "feature": FINAL_FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    imp_df.to_csv(OUTPUT_DIR / "final_feature_importance.csv", index=False)
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
