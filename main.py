"""
Point d'entrée unique — baseline + audits gaitpdb.

Usage : python main.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from project.audit import (
    feature_importance_report,
    study_effect_check,
    updrsm_robustness_check,
)
from project.features import FEATURE_COLS, build_feature_matrix

RANDOM_STATE = 42
SESSION = "01"
_REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = str(_REPO_ROOT / "output")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Phase 1 — Baseline
# ---------------------------------------------------------------------------


def run_phase1(df: pd.DataFrame) -> list[dict]:
    _sep("Baseline RF — PD vs CO + UPDRSM")

    # --- Tâche 1 : PD vs CO ---
    clf_df = df.dropna(subset=FEATURE_COLS).copy()
    clf_df["label"] = (clf_df["group"] == "PD").astype(int)
    X = clf_df[FEATURE_COLS].values
    y = clf_df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    clf_row = {
        "task": "PD_vs_CO",
        "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1_weighted": round(f1_score(y_te, y_pred, average="weighted"), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "mae": np.nan, "rmse": np.nan, "r2": np.nan,
        "spearman_r": np.nan, "spearman_p": np.nan,
    }
    print("Tache 1 : PD vs CO")
    for k, v in clf_row.items():
        print(f"  {k}: {v}")

    # --- Tâche 2 : régression UPDRSM ---
    reg_df = df[
        (df["group"] == "PD") & df["UPDRSM"].notna()
    ].dropna(subset=FEATURE_COLS).copy()
    X_r = reg_df[FEATURE_COLS].values
    y_r = reg_df["UPDRSM"].values

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_r, y_r, test_size=0.25, random_state=RANDOM_STATE
    )
    reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    reg.fit(X_tr_r, y_tr_r)
    y_pred_r = reg.predict(X_te_r)
    sp_r, sp_p = spearmanr(y_te_r, y_pred_r)

    reg_row = {
        "task": "UPDRSM_regression",
        "n_train": len(X_tr_r), "n_test": len(X_te_r),
        "accuracy": np.nan, "f1_weighted": np.nan, "roc_auc": np.nan,
        "mae": round(float(mean_absolute_error(y_te_r, y_pred_r)), 4),
        "rmse": round(_rmse(y_te_r, y_pred_r), 4),
        "r2": round(_r2(y_te_r, y_pred_r), 4),
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
    }
    print("\nTache 2 : Regression UPDRSM (PD uniquement)")
    for k, v in reg_row.items():
        print(f"  {k}: {v}")

    return [clf_row, reg_row]


# ---------------------------------------------------------------------------
# Phase 1.5 — Audits
# ---------------------------------------------------------------------------


def run_phase15(df: pd.DataFrame) -> None:
    _sep("Audit — Effet study")

    study_only, loso_df = study_effect_check(df, FEATURE_COLS)
    print("LR avec study one-hot uniquement :")
    for k, v in study_only.items():
        print(f"  {k}: {v}")

    print("\nLOSO RF (toutes features, 1 etude tenue hors entrainement) :")
    print(loso_df.to_string(index=False))
    loso_df.to_csv(os.path.join(OUTPUT_DIR, "loso_results.csv"), index=False)
    pd.DataFrame([{"analysis": "study_only", **study_only}]).to_csv(
        os.path.join(OUTPUT_DIR, "study_effect.csv"), index=False
    )

    _sep("Audit — Importance des features")

    rf_imp, perm_imp = feature_importance_report(df, FEATURE_COLS, OUTPUT_DIR)

    print("\nTop 10 RF (MDI) :")
    print(rf_imp.head(10).to_string(index=False))

    print("\nTop 10 Permutation Importance :")
    print(perm_imp.head(10).to_string(index=False))

    # Position des features d'asymétrie
    asym_features = [
        "mean_asym", "std_asym", "diff_auc",
        "diff_peak", "ratio_auc_L_over_R", "mean_abs_diff",
    ]
    print("\nRang des features d'asymetrie (permutation) :")
    for feat in asym_features:
        row = perm_imp[perm_imp["feature"] == feat]
        if not row.empty:
            rank = row.index[0] + 1
            mean_val = row["perm_mean"].values[0]
            print(f"  {feat:25s}  rang {rank:2d}  perm_mean={mean_val:.4f}")

    rf_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_rf.csv"), index=False)
    perm_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_perm.csv"), index=False)

    _sep("Audit — Robustesse regression UPDRSM")

    rob_df = updrsm_robustness_check(df, FEATURE_COLS)
    print(rob_df.to_string(index=False))
    rob_df.to_csv(os.path.join(OUTPUT_DIR, "updrsm_robustness.csv"), index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Session : {SESSION}  (marche normale uniquement)")
    df = build_feature_matrix(session=SESSION)
    print(f"Sujets charges : {len(df)}")

    dist = df.groupby(["study", "group"]).size().unstack(fill_value=0)
    print("\nDistribution par etude :")
    print(dist.to_string())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    baseline_rows = run_phase1(df)
    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(OUTPUT_DIR, "baseline_metrics.csv"), index=False
    )

    run_phase15(df)

    # Résumé final
    _sep("RESUME FINAL")
    b = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_metrics.csv"))
    loso = pd.read_csv(os.path.join(OUTPUT_DIR, "loso_results.csv"))
    study_eff = pd.read_csv(os.path.join(OUTPUT_DIR, "study_effect.csv"))
    perm = pd.read_csv(os.path.join(OUTPUT_DIR, "feature_importance_perm.csv"))
    rob = pd.read_csv(os.path.join(OUTPUT_DIR, "updrsm_robustness.csv"))

    print(f"\nAUC baseline RF (PD vs CO)    : {b.loc[0, 'roc_auc']}")
    print(f"AUC avec study seul (LR)       : {study_eff.loc[0, 'roc_auc']}")
    print(f"\nLOSO AUC par etude :")
    for _, row in loso.iterrows():
        print(f"  Test {row['held_out_study']} : AUC={row['roc_auc']}  acc={row['accuracy']}")

    print(f"\nTop 10 features (permutation importance) :")
    for _, row in perm.head(10).iterrows():
        print(f"  {row['feature']:25s}  {row['perm_mean']:.4f} +/- {row['perm_std']:.4f}")

    print(f"\nRobustesse UPDRSM (R2 sur 5 seeds) :")
    r2_vals = rob["r2"].tolist()
    print(f"  {r2_vals}  -> moy={np.mean(r2_vals):.3f}")

    print("\nOutputs sauvegardes dans output/")


if __name__ == "__main__":
    main()
