"""
Comparaison structurée des familles de features — classification PD vs CO.

Les quatre familles sont évaluées sur le même split train/test (mêmes sujets,
même seed) pour que les métriques soient directement comparables.

Absence de NaN vérifiée : 0 NaN sur les 165 sujets session 01.

Usage depuis la racine du dépôt :
    python -m project.compare_families
    python project/compare_families.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from project.features import (
    ASYMMETRY_FEATURES,
    COMBINED_FEATURES,
    FEATURE_COLS,
    TEMPORAL_FEATURES,
    build_feature_matrix,
)

RANDOM_STATE = 42
SESSION = "01"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"

FAMILIES: dict[str, list[str]] = {
    "temporal_only": TEMPORAL_FEATURES,
    "asymmetry_only": ASYMMETRY_FEATURES,
    "combined": COMBINED_FEATURES,
    "all_features": FEATURE_COLS,
}


def _nan_report(df: pd.DataFrame) -> None:
    print("Verification NaN par famille (session 01) :")
    for name, cols in FAMILIES.items():
        n_valid = df.dropna(subset=cols).shape[0]
        n_nan = len(df) - n_valid
        flag = "  [OK]" if n_nan == 0 else f"  [!] {n_nan} sujets exclus"
        print(f"  {name:20s}: {n_valid}/{len(df)} valides{flag}")


def run_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    base = df.dropna(subset=FEATURE_COLS).copy()
    base["label"] = (base["group"] == "PD").astype(int)
    y = base["label"].values

    idx = np.arange(len(base))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    y_tr, y_te = y[idx_tr], y[idx_te]

    rows = []
    importances: dict[str, pd.Series] = {}

    for name, cols in FAMILIES.items():
        X = base[cols].values
        X_tr, X_te = X[idx_tr], X[idx_te]

        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        y_prob = rf.predict_proba(X_te)[:, 1]

        importances[name] = pd.Series(
            rf.feature_importances_, index=cols
        ).sort_values(ascending=False)

        rows.append({
            "family": name,
            "n_features": len(cols),
            "n_train": len(idx_tr),
            "n_test": len(idx_te),
            "accuracy": round(accuracy_score(y_te, y_pred), 4),
            "f1_weighted": round(f1_score(y_te, y_pred, average="weighted"), 4),
            "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        })

    return pd.DataFrame(rows), importances


def main() -> None:
    print(f"Session : {SESSION}  (marche normale uniquement)")
    df = build_feature_matrix(session=SESSION)
    print(f"Sujets charges : {len(df)}\n")

    _nan_report(df)
    print()

    results, importances = run_comparison(df)

    print("=" * 65)
    print("  Comparaison des familles de features — PD vs CO")
    print("=" * 65)
    print(results[["family", "n_features", "accuracy", "f1_weighted", "roc_auc"]].to_string(index=False))
    print()

    asym_set = set(ASYMMETRY_FEATURES)
    temp_set = set(TEMPORAL_FEATURES)
    print("Top features par famille (RF MDI) :")
    for name, imp in importances.items():
        parts = []
        for feat, val in imp.head(6).items():
            tag = "[A]" if feat in asym_set else "[T]" if feat in temp_set else "[ ]"
            parts.append(f"{tag}{feat}({val:.3f})")
        print(f"  {name:20s}: {', '.join(parts)}")
    print("\n  [A]=asymetrie  [T]=temporel  [ ]=force brute")

    _OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = _OUTPUT_DIR / "feature_family_comparison.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResultats -> {out_path}")


if __name__ == "__main__":
    main()
