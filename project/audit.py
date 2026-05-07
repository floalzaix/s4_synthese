"""
Phase 1.5 — Audits de sécurisation baseline.

Trois modules indépendants :
  study_effect_check        — LR study-seul + LOSO RF
  feature_importance_report — importances RF + permutation + plot PNG
  updrsm_robustness_check   — régression UPDRSM sur plusieurs seeds
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_RS = 42
_DEFAULT_OUTPUT = str(Path(__file__).resolve().parent.parent / "output")


# ---------------------------------------------------------------------------
# 1. Effet "study"
# ---------------------------------------------------------------------------


def study_effect_check(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, pd.DataFrame]:
    """
    Deux vérifications :
    a) LR avec `study` one-hot uniquement → est-ce que l'étude prédit PD/CO ?
    b) LOSO RF (all features) : train sur 2 études, test sur la 3ème.

    Returns (study_only_metrics, loso_df).
    """
    df = df.dropna(subset=feature_cols).copy()
    df["label"] = (df["group"] == "PD").astype(int)
    studies = sorted(df["study"].unique())

    # --- a) LR sur study one-hot ---
    study_dummies = pd.get_dummies(df["study"], prefix="study").astype(float)
    X_s = study_dummies.values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.25, stratify=y, random_state=_RS
    )
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=500, random_state=_RS)
    lr.fit(scaler.fit_transform(X_tr), y_tr)
    y_pred_lr = lr.predict(scaler.transform(X_te))
    y_prob_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]

    study_only = {
        "accuracy": round(accuracy_score(y_te, y_pred_lr), 4),
        "f1_weighted": round(f1_score(y_te, y_pred_lr, average="weighted"), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob_lr), 4),
    }

    # --- b) LOSO RF ---
    X_all = df[feature_cols].values
    rows = []
    for held_out in studies:
        tr_mask = (df["study"] != held_out).values
        te_mask = (df["study"] == held_out).values
        X_tr_l, y_tr_l = X_all[tr_mask], y[tr_mask]
        X_te_l, y_te_l = X_all[te_mask], y[te_mask]

        rf = RandomForestClassifier(n_estimators=200, random_state=_RS)
        rf.fit(X_tr_l, y_tr_l)
        y_pred_l = rf.predict(X_te_l)
        y_prob_l = rf.predict_proba(X_te_l)[:, 1]

        rows.append({
            "held_out_study": held_out,
            "n_test": int(te_mask.sum()),
            "accuracy": round(accuracy_score(y_te_l, y_pred_l), 4),
            "f1_weighted": round(f1_score(y_te_l, y_pred_l, average="weighted"), 4),
            "roc_auc": round(roc_auc_score(y_te_l, y_prob_l), 4),
        })

    return study_only, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Importance des features
# ---------------------------------------------------------------------------


def feature_importance_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = _DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entraîne un RF (75/25 stratifié) et calcule :
    - importances intégrées (MDI)
    - permutation importance sur le jeu de test (30 répétitions)
    - sauvegarde un plot PNG

    Returns (rf_imp_df, perm_imp_df).
    """
    df = df.dropna(subset=feature_cols).copy()
    df["label"] = (df["group"] == "PD").astype(int)
    X = df[feature_cols].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_RS
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=_RS)
    rf.fit(X_tr, y_tr)

    rf_imp = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf.feature_importances_,
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    perm = permutation_importance(
        rf, X_te, y_te, n_repeats=30, random_state=_RS, n_jobs=1
    )
    perm_imp = pd.DataFrame({
        "feature": feature_cols,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)

    _save_importance_plot(rf_imp, perm_imp, output_dir)

    return rf_imp, perm_imp


def _save_importance_plot(
    rf_imp: pd.DataFrame,
    perm_imp: pd.DataFrame,
    output_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    top_rf = rf_imp.head(15).iloc[::-1]
    axes[0].barh(top_rf["feature"], top_rf["rf_importance"], color="steelblue")
    axes[0].set_title("RF Importance (MDI) — top 15")
    axes[0].set_xlabel("Mean decrease impurity")

    top_pm = perm_imp.head(15).iloc[::-1]
    axes[1].barh(
        top_pm["feature"],
        top_pm["perm_mean"],
        xerr=top_pm["perm_std"],
        color="darkorange",
        ecolor="gray",
        capsize=3,
    )
    axes[1].set_title("Permutation Importance (test set) — top 15")
    axes[1].set_xlabel("Mean decrease in accuracy")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")


# ---------------------------------------------------------------------------
# 3. Robustesse régression UPDRSM
# ---------------------------------------------------------------------------


def updrsm_robustness_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    seeds: tuple[int, ...] = (42, 0, 123, 7, 1),
) -> pd.DataFrame:
    """
    Répète la régression UPDRSM (PD, UPDRSM disponible) sur plusieurs seeds.
    Retourne un DataFrame avec MAE, RMSE, R², Spearman par seed.
    """
    reg_df = df[
        (df["group"] == "PD") & df["UPDRSM"].notna()
    ].dropna(subset=feature_cols).copy()

    X = reg_df[feature_cols].values
    y = reg_df["UPDRSM"].values

    rows = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )
        reg = RandomForestRegressor(n_estimators=200, random_state=seed)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        ss_res = np.sum((y_pred - y_te) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        sp_r, sp_p = spearmanr(y_te, y_pred)

        rows.append({
            "seed": seed,
            "n_test": len(X_te),
            "mae": round(float(mean_absolute_error(y_te, y_pred)), 3),
            "rmse": round(float(np.sqrt(np.mean((y_pred - y_te) ** 2))), 3),
            "r2": round(float(1 - ss_res / (ss_tot + 1e-9)), 3),
            "spearman_r": round(float(sp_r), 3),
            "spearman_p": round(float(sp_p), 3),
        })

    return pd.DataFrame(rows)
