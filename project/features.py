"""
[ROLE]
Ce fichier contient les fonctions d'extraction de caractéristiques (features) à partir des signaux de marche.

[RESPONSIBILITY]
- Calculer des caractéristiques de force et d'asymétrie.
- Calculer des caractéristiques temporelles (cadence, intervalles).
- Segmenter les pas (stub pour phase 4).
- Construire la matrice de caractéristiques pour l'ensemble du dataset.

[INPUTS]
- DataFrame des signaux temporels (provenant de project.load).

[OUTPUTS]
- Dictionnaires de caractéristiques.
- DataFrame pandas (matrice de caractéristiques).

[ASSUMPTIONS]
- Signaux échantillonnés à 100 Hz.
- Les colonnes total_L et total_R sont présentes.

[RISKS]
- La détection de pics peut être sensible au bruit si le signal n'est pas filtré.
- Les caractéristiques agrégées peuvent masquer des phénomènes dynamiques fins.

[DEPENDENCIES]
- numpy
- pandas
- scipy.signal
- project.load
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from project.load import load_dataset_index, load_signal_file

_EPS = 1e-9
_FS = 100  # Hz
_MIN_PEAK_DIST = 40  # samples = 0.4 s, intervalle minimal entre deux pas
_PEAK_HEIGHT_RATIO = 0.20
_STANCE_THRESHOLD_RATIO = 0.05  # 5% of max signal as stance threshold

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

STEP_FEATURES: list[str] = [
    "n_steps_L_seg",
    "n_steps_R_seg",
    "mean_stance_L",
    "mean_stance_R",
    "mean_swing_L",
    "mean_swing_R",
    "cv_stance_L",
    "cv_stance_R",
    "cv_swing_L",
    "cv_swing_R",
    "asym_stance",
    "asym_swing",
    "asym_auc_steps",
]

# Toutes les features calculées par extract_features (20 colonnes + step features)
FEATURE_COLS: list[str] = [
    "mean_L",
    "std_L",
    "mean_R",
    "std_R",
    "mean_asym",
    "std_asym",
    "mean_abs_diff",
    "peak_L",
    "peak_R",
    "diff_peak",
    "auc_L",
    "auc_R",
    "diff_auc",
    "ratio_auc_L_over_R",
    "n_steps",
    "cadence_spm",
    "mean_interval_L",
    "cv_interval_L",
    "mean_interval_R",
    "cv_interval_R",
] + STEP_FEATURES

META_COLS: list[str] = ["subject_id", "group", "study", "UPDRSM"]


def _force_features(L: np.ndarray, R: np.ndarray) -> dict:
    """
    @brief Calcule les caractéristiques basées sur la force et l'asymétrie.
    @param L Signal de force total jambe gauche.
    @param R Signal de force total jambe droite.
    @return dict Caractéristiques de force calculées.
    """
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
    """
    @brief Calcule les caractéristiques temporelles via détection de pics.
    @param L Signal de force total jambe gauche.
    @param R Signal de force total jambe droite.
    @param duration Durée totale de l'enregistrement en secondes.
    @return dict Caractéristiques temporelles calculées.
    """

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


def segment_steps(x: np.ndarray) -> list[tuple[int, int]]:
    """
    @brief Identifie les segments (indices début, fin) des phases d'appui.
    @param x Signal de force (L ou R).
    @return list[tuple[int, int]] Liste des segments (start, end).
    """
    max_val = np.max(x)
    if max_val <= 0:
        return []
    threshold = max_val * _STANCE_THRESHOLD_RATIO
    above = x > threshold
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(x) - 1)

    segments = []
    for s, e in zip(starts, ends):
        # Un pas doit durer au moins 0.1s (10 samples @ 100Hz)
        if e - s >= 10:
            segments.append((s, e))
    return segments


def extract_step_features(sig: pd.DataFrame) -> dict:
    """
    @brief Extrait des caractéristiques par pas (stance/swing).

    Un pas (phase d'appui / stance) est détecté lorsque le signal dépasse
    un seuil fixé à 5% du pic maximum du signal. La phase d'oscillation (swing)
    est définie comme l'intervalle entre deux phases d'appui successives.

    @param sig DataFrame contenant les signaux temporels.
    @return dict Caractéristiques physiologiques agrégées sur les pas.
    """
    L = sig["total_L"].values
    R = sig["total_R"].values

    seg_L = segment_steps(L)
    seg_R = segment_steps(R)

    def _calc_stance(segments: list[tuple[int, int]]) -> np.ndarray:
        if not segments:
            return np.array([])
        return np.array([e - s for s, e in segments]) / _FS

    def _calc_swing(segments: list[tuple[int, int]]) -> np.ndarray:
        if len(segments) < 2:
            return np.array([])
        swings = []
        for i in range(len(segments) - 1):
            s = segments[i + 1][0] - segments[i][1]
            if 0 < s < 200:  # Filtre : swing max 2s (200 samples)
                swings.append(s)
        return np.array(swings) / _FS

    def _calc_auc(x: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
        if not segments:
            return np.array([])
        return np.array([float(np.trapezoid(x[s:e])) for s, e in segments])

    stance_L = _calc_stance(seg_L)
    stance_R = _calc_stance(seg_R)
    swing_L = _calc_swing(seg_L)
    swing_R = _calc_swing(seg_R)
    auc_L_steps = _calc_auc(L, seg_L)
    auc_R_steps = _calc_auc(R, seg_R)

    def safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else np.nan

    def safe_cv(arr: np.ndarray) -> float:
        return float(np.std(arr) / (np.mean(arr) + _EPS)) if len(arr) > 1 else np.nan

    m_stance_L = safe_mean(stance_L)
    m_stance_R = safe_mean(stance_R)
    m_swing_L = safe_mean(swing_L)
    m_swing_R = safe_mean(swing_R)
    m_auc_L = safe_mean(auc_L_steps)
    m_auc_R = safe_mean(auc_R_steps)

    asym_stance = (
        np.abs(m_stance_L - m_stance_R) / (m_stance_L + m_stance_R + _EPS)
        if not np.isnan(m_stance_L) and not np.isnan(m_stance_R)
        else np.nan
    )
    asym_swing = (
        np.abs(m_swing_L - m_swing_R) / (m_swing_L + m_swing_R + _EPS)
        if not np.isnan(m_swing_L) and not np.isnan(m_swing_R)
        else np.nan
    )
    asym_auc = (
        np.abs(m_auc_L - m_auc_R) / (m_auc_L + m_auc_R + _EPS)
        if not np.isnan(m_auc_L) and not np.isnan(m_auc_R)
        else np.nan
    )

    return {
        "n_steps_L_seg": float(len(seg_L)),
        "n_steps_R_seg": float(len(seg_R)),
        "mean_stance_L": m_stance_L,
        "mean_stance_R": m_stance_R,
        "mean_swing_L": m_swing_L,
        "mean_swing_R": m_swing_R,
        "cv_stance_L": safe_cv(stance_L),
        "cv_stance_R": safe_cv(stance_R),
        "cv_swing_L": safe_cv(swing_L),
        "cv_swing_R": safe_cv(swing_R),
        "asym_stance": asym_stance,
        "asym_swing": asym_swing,
        "asym_auc_steps": asym_auc,
    }


def extract_features(sig: pd.DataFrame) -> dict:
    """
    @brief Extrait toutes les caractéristiques d'un signal donné.
    @param sig DataFrame contenant les signaux temporels.
    @return dict Dictionnaire complet des caractéristiques.
    """
    L = sig["total_L"].values
    R = sig["total_R"].values
    duration = float(sig["time"].iloc[-1] - sig["time"].iloc[0])
    feats: dict = {}
    feats.update(_force_features(L, R))
    feats.update(_temporal_features(L, R, duration))
    feats.update(extract_step_features(sig))
    return feats


def build_feature_matrix(
    root_dir=None,
    session: str = "01",
) -> pd.DataFrame:
    """
    @brief Construit la matrice features pour la session donnée.

    Retourne un DataFrame (META_COLS + FEATURE_COLS), 1 ligne = 1 sujet.
    Seules les entrées avec has_signal=True sont incluses.

    @param root_dir Chemin optionnel vers le dataset.
    @param session Identifiant de la session (défaut "01").
    @return pd.DataFrame Matrice de caractéristiques.
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
