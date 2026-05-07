"""
Features agrégées sur l'enregistrement entier (pas de segmentation par cycle).

Features force/asymétrie : dérivées de total_L et total_R (sommes des 8 capteurs).
Features temporelles : détection de pics de charge via scipy.signal.find_peaks
(hauteur ≥ 20 % du max du signal, distance ≥ 0.4 s).
Fiables pour marche normale en ligne droite (session 01) ; à ne pas utiliser
telles quelles pour les sessions dual-task ou RAS sans réévaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from project.load import load_dataset_index, load_signal_file

_EPS = 1e-9
_FS = 100            # Hz
_MIN_PEAK_DIST = 40  # samples = 0.4 s, intervalle minimal entre deux pas
_PEAK_HEIGHT_RATIO = 0.20

TEMPORAL_FEATURES: list[str] = [
    "n_steps",
    "cadence_spm",
    "mean_interval_L",
    "cv_interval_L",
    "mean_interval_R",
    "cv_interval_R",
]

ASYMMETRY_FEATURES: list[str] = [
    "mean_asym",
    "std_asym",
    "mean_abs_diff",
    "diff_peak",
    "diff_auc",
    "ratio_auc_L_over_R",
]

# Union asymétrie + temporel (ordre stable : asymétrie d'abord)
COMBINED_FEATURES: list[str] = ASYMMETRY_FEATURES + TEMPORAL_FEATURES

# Toutes les features calculées par extract_features (20 colonnes)
FEATURE_COLS: list[str] = [
    "mean_L", "std_L", "mean_R", "std_R",
    "mean_asym", "std_asym", "mean_abs_diff",
    "peak_L", "peak_R", "diff_peak",
    "auc_L", "auc_R", "diff_auc", "ratio_auc_L_over_R",
    "n_steps", "cadence_spm",
    "mean_interval_L", "cv_interval_L",
    "mean_interval_R", "cv_interval_R",
]

META_COLS: list[str] = ["subject_id", "group", "study", "UPDRSM"]


def _force_features(L: np.ndarray, R: np.ndarray) -> dict:
    asym = (R - L) / (R + L + _EPS)
    auc_L = float(np.trapezoid(L))
    auc_R = float(np.trapezoid(R))
    return {
        "mean_L": float(L.mean()),
        "std_L": float(L.std()),
        "mean_R": float(R.mean()),
        "std_R": float(R.std()),
        "mean_asym": float(asym.mean()),
        "std_asym": float(asym.std()),
        "mean_abs_diff": float(np.abs(L - R).mean()),
        "peak_L": float(L.max()),
        "peak_R": float(R.max()),
        "diff_peak": float(R.max() - L.max()),
        "auc_L": auc_L,
        "auc_R": auc_R,
        "diff_auc": auc_R - auc_L,
        "ratio_auc_L_over_R": auc_L / (auc_R + _EPS),
    }


def _temporal_features(L: np.ndarray, R: np.ndarray, duration: float) -> dict:
    def _peaks(x: np.ndarray) -> np.ndarray:
        h = x.max() * _PEAK_HEIGHT_RATIO
        idx, _ = find_peaks(x, height=h, distance=_MIN_PEAK_DIST)
        return idx

    def _interval_stats(idx: np.ndarray) -> tuple[float, float]:
        if len(idx) < 2:
            return np.nan, np.nan
        ivs = np.diff(idx) / _FS  # secondes
        return float(ivs.mean()), float(ivs.std() / (ivs.mean() + _EPS))

    peaks_L = _peaks(L)
    peaks_R = _peaks(R)
    n_steps = len(peaks_L) + len(peaks_R)
    cadence = (n_steps / duration * 60) if duration > 0 else np.nan

    mi_L, cv_L = _interval_stats(peaks_L)
    mi_R, cv_R = _interval_stats(peaks_R)

    return {
        "n_steps": n_steps,
        "cadence_spm": cadence,
        "mean_interval_L": mi_L,
        "cv_interval_L": cv_L,
        "mean_interval_R": mi_R,
        "cv_interval_R": cv_R,
    }


def extract_features(sig: pd.DataFrame) -> dict:
    L = sig["total_L"].values
    R = sig["total_R"].values
    duration = float(sig["time"].iloc[-1] - sig["time"].iloc[0])
    feats: dict = {}
    feats.update(_force_features(L, R))
    feats.update(_temporal_features(L, R, duration))
    return feats


def build_feature_matrix(
    root_dir=None,
    session: str = "01",
) -> pd.DataFrame:
    """
    Construit la matrice features pour la session donnée.

    Retourne un DataFrame (META_COLS + FEATURE_COLS), 1 ligne = 1 sujet.
    Seules les entrées avec has_signal=True sont incluses.
    """
    index = load_dataset_index(root_dir)

    # Session 01 = marche normale (décision Phase 1 : pas de mélange de protocoles)
    mask = (index["session"] == session) & index["has_signal"]
    subset = index[mask].copy()

    records = []
    for _, row in subset.iterrows():
        sig = load_signal_file(row["filepath"])
        feats = extract_features(sig)
        feats["subject_id"] = row["subject_id"]
        feats["group"] = row["group"]
        feats["study"] = row["study"]
        feats["UPDRSM"] = row.get("UPDRSM", np.nan)
        records.append(feats)

    return pd.DataFrame(records)[META_COLS + FEATURE_COLS]
