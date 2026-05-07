"""
Baseline classification PD vs CO + régression UPDRSM.

Session 01 uniquement (marche normale, 1 ligne = 1 sujet).
Split stratifié 75/25 par sujet ; pas de fuite de données.

Usage depuis la racine du dépôt :
    python -m project.train
    python project/train.py
"""
from __future__ import annotations

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

from project.features import FEATURE_COLS, build_feature_matrix

RANDOM_STATE = 42
SESSION = "01"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"
OUTPUT_FILE = str(_OUTPUT_DIR / "baseline_metrics.csv")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


def main() -> None:
    print(f"Session : {SESSION}  (marche normale uniquement)")
    df = build_feature_matrix(session=SESSION)
    print(f"Sujets charges : {len(df)}\n")

    dist = df.groupby(["study", "group"]).size().unstack(fill_value=0)
    print("Distribution par etude :")
    print(dist.to_string())
    print()

    # Tache 1 : PD vs CO
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

    clf_row: dict = {
        "task": "PD_vs_CO",
        "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1_weighted": round(f1_score(y_te, y_pred, average="weighted"), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "mae": np.nan, "rmse": np.nan, "r2": np.nan,
        "spearman_r": np.nan, "spearman_p": np.nan,
    }
    print("Classification PD vs CO :")
    for k, v in clf_row.items():
        print(f"  {k}: {v}")
    print()

    # Tache 2 : regression UPDRSM (PD uniquement)
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

    reg_row: dict = {
        "task": "UPDRSM_regression",
        "n_train": len(X_tr_r), "n_test": len(X_te_r),
        "accuracy": np.nan, "f1_weighted": np.nan, "roc_auc": np.nan,
        "mae": round(float(mean_absolute_error(y_te_r, y_pred_r)), 4),
        "rmse": round(_rmse(y_te_r, y_pred_r), 4),
        "r2": round(_r2(y_te_r, y_pred_r), 4),
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
    }
    print("Regression UPDRSM (PD uniquement) :")
    for k, v in reg_row.items():
        print(f"  {k}: {v}")
    print()

    _OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame([clf_row, reg_row]).to_csv(OUTPUT_FILE, index=False)
    print(f"Resultats -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
