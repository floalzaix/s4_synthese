"""
[ROLE]
Améliorations pragmatiques des performances step-level pour GaitPDB.

[AXES D'AMÉLIORATION]
  1. Optimisation du seuil de décision sujet-level (Youden / F1-max / accuracy).
  2. Features enrichies : rise_time, fall_time, impulse_ratio, normalisation sujet.
  3. Traitement du signal : filtrage Butterworth 10 Hz avant extraction features.
  4. Stacking léger RF + LogReg → LogReg niveau sujet (OOF meta-features, LOOCV).

[CONTRAINTES]
  - Pas de modification de la segmentation ni de StepDataset.
  - Pas de données brutes supplémentaires.
  - Compatibilité StratifiedGroupKFold stricte (pas de fuite de données).

[OUTPUTS]
  output/etude_du_pas/threshold.json
  output/etude_du_pas/features_enriched.csv
  output/etude_du_pas/stacking_results.csv
  output/etude_du_pas/baseline_plus_improvements.csv
  output/etude_du_pas/figures/improvements/threshold_curve.png
  output/etude_du_pas/figures/improvements/signal_filter_comparison.png
  output/etude_du_pas/figures/improvements/performance_comparison.png
  output/etude_du_pas/figures/improvements/stacking_comparison.png

[DÉPENDANCES]
  - scipy, scikit-learn, matplotlib, seaborn, numpy, pandas
  - gaitpdb.model_step_comparison, gaitpdb.config, gaitpdb.viz.utils
  - gaitpdb.step_dataset, gaitpdb.load
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from gaitpdb.config import OUTPUT_DIR, REPO_ROOT, RANDOM_STATE
from gaitpdb.load import load_signal_file

# Shim numpy trapz → trapezoid (supprimé en NumPy 2.2+)
try:
    _np_trapz = np.trapezoid
except AttributeError:
    _np_trapz = np.trapz  # type: ignore[attr-defined]
from gaitpdb.model_step_comparison import (
    _EPS,
    _N_SPLITS,
    _TAB_FEATURE_NAMES,
    build_step_features,
    get_classifiers,
    run_classifier_cv,
)
from gaitpdb.step_dataset import StepDataset, _SENSOR_COLS
from gaitpdb.viz.utils import save_fig, setup_style

setup_style()

# ---------------------------------------------------------------------------
# Chemins de sortie
# ---------------------------------------------------------------------------

_OUT_DIR: Path = OUTPUT_DIR / "etude_du_pas"
_FIG_DIR: Path = _OUT_DIR / "figures" / "improvements"

# ---------------------------------------------------------------------------
# Paramètres du filtrage Butterworth
# Cutoff 10 Hz : standard pour les signaux de pression plantaire (100 Hz).
# Ordre 4 zero-phase (filtfilt) : atténuation forte sans déphasage.
# Référence : Bartlett et al. 2007, Journal of Biomechanics.
# ---------------------------------------------------------------------------

_FS: int = 100          # Hz — fréquence d'échantillonnage GaitPDB
_FILTER_CUTOFF: float = 10.0   # Hz — passe-bas plantaire
_FILTER_ORDER: int = 4

# Features enrichies (base + nouvelles)
_ENRICH_FEATURE_NAMES: list[str] = [
    # Originales
    "duration_s",
    "peak_force",
    "auc_norm",
    "foot_L",
    "stride_asym_peak",
    "stride_asym_duration",
    "stride_asym_auc",
    # Nouvelles — forme du pas
    "rise_time_s",
    "fall_time_s",
    "rise_fall_ratio",
    "impulse_front_ratio",
    "peak_force_subj_norm",
    # Nouvelles — asymétries enrichies
    "stride_asym_rise",
    "stride_asym_fall",
    "stride_asym_impulse",
]


# Features capteurs individuels (P6)
_SENSOR_MEAN_NAMES: list[str] = [f"mean_{c}" for c in _SENSOR_COLS]

_SENSOR_RATIO_NAMES: list[str] = [
    "forefoot_ratio_L",
    "forefoot_ratio_R",
    "forefoot_asym",
    "heel_dominance",
]

_SENSOR_ASYM_NAMES: list[str] = [f"sensor_asym_{i}" for i in range(1, 9)]

_SENSOR_ENTROPY_NAMES: list[str] = ["entropy_L", "entropy_R"]

_SENSOR_FEATURE_NAMES: list[str] = (
    _SENSOR_MEAN_NAMES + _SENSOR_RATIO_NAMES + _SENSOR_ASYM_NAMES + _SENSOR_ENTROPY_NAMES
)

_ALL_TAB_FEATURE_NAMES: list[str] = _ENRICH_FEATURE_NAMES + _SENSOR_FEATURE_NAMES


# ===========================================================================
# 1. TRAITEMENT DU SIGNAL
# ===========================================================================


def _butter_filter(signal: np.ndarray) -> np.ndarray:
    """
    Filtre passe-bas Butterworth zero-phase.
    Cutoff 10 Hz @ 100 Hz — standard pour la pression plantaire.
    Retourne le signal original si la fenêtre est trop courte pour filtrer.
    """
    min_len = 3 * _FILTER_ORDER
    if len(signal) < min_len:
        return signal.copy()
    nyq = _FS / 2.0
    b, a = butter(_FILTER_ORDER, _FILTER_CUTOFF / nyq, btype="low")
    return filtfilt(b, a, signal).astype(signal.dtype)


def _load_filtered_signals(steps_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Charge et filtre les signaux total_L / total_R pour chaque fichier source.
    Retour : {source_file: {"L": array_filtré, "R": array_filtré}}
    Cache : chaque fichier est chargé et filtré une seule fois.
    """
    cache: dict[str, dict[str, np.ndarray]] = {}
    source_files = steps_df["source_file"].unique()

    for sf in source_files:
        try:
            sig = load_signal_file(REPO_ROOT / Path(sf))
        except Exception as exc:
            warnings.warn(f"Impossible de charger {sf} : {exc}", RuntimeWarning)
            continue
        cache[sf] = {
            "L": _butter_filter(sig["total_L"].values.astype(np.float32)),
            "R": _butter_filter(sig["total_R"].values.astype(np.float32)),
        }

    return cache


# ===========================================================================
# 2. FEATURES ENRICHIES
# ===========================================================================


def _step_shape_features(window: np.ndarray) -> dict:
    """
    Extrait les features de forme du pas à partir du signal filtré.

    rise_time_s        : temps du début au pic (phase de chargement).
    fall_time_s        : temps du pic à la fin (phase de déchargement).
    rise_fall_ratio    : rise/fall — > 1 si chargement lent, < 1 si déchargement lent.
    impulse_front_ratio: AUC de la première moitié / AUC totale — dominance de la
                         phase de chargement. Valeur normale ~0.45–0.55.
    """
    n = len(window)
    if n == 0:
        return {
            "rise_time_s": np.nan,
            "fall_time_s": np.nan,
            "rise_fall_ratio": np.nan,
            "impulse_front_ratio": np.nan,
        }

    peak_idx = int(np.argmax(window))
    rise_time_s = peak_idx / _FS
    fall_time_s = (n - 1 - peak_idx) / _FS

    # AUC des deux moitiés (trapèze discret)
    half = n // 2
    auc_total = float(_np_trapz(window)) + _EPS
    auc_front = float(_np_trapz(window[:half])) if half > 0 else 0.0
    impulse_front_ratio = auc_front / auc_total

    return {
        "rise_time_s": rise_time_s,
        "fall_time_s": fall_time_s,
        "rise_fall_ratio": rise_time_s / (fall_time_s + _EPS),
        "impulse_front_ratio": impulse_front_ratio,
    }


def _compute_extended_asymmetries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les asymétries intra-foulée pour les nouvelles features de forme.

    Mêmes principes que _compute_stride_asymmetries() dans model_step_comparison.py :
    - Jointement par (subject_id, session, stride_id)
    - Asymétrie = |L - R| / (L + R + eps)
    - NaN pour les pas non appariés
    """
    paired = df[df["stride_id"].notna()].copy()
    if paired.empty:
        return pd.DataFrame(columns=["stride_asym_rise", "stride_asym_fall", "stride_asym_impulse"])

    paired["stride_id_int"] = paired["stride_id"].astype(int)
    records: list[dict] = []
    key = ["subject_id", "session", "stride_id_int"]

    for _, grp in paired.groupby(key):
        if set(grp["foot"].unique()) != {"L", "R"}:
            continue
        l = grp[grp["foot"] == "L"].iloc[0]
        r = grp[grp["foot"] == "R"].iloc[0]

        def _asym(a, b):
            return abs(a - b) / (a + b + _EPS)

        for step_id in (l["step_id"], r["step_id"]):
            records.append({
                "step_id": step_id,
                "stride_asym_rise":    _asym(l["rise_time_s"],         r["rise_time_s"]),
                "stride_asym_fall":    _asym(l["fall_time_s"],         r["fall_time_s"]),
                "stride_asym_impulse": _asym(l["impulse_front_ratio"], r["impulse_front_ratio"]),
            })

    if not records:
        return pd.DataFrame(columns=["stride_asym_rise", "stride_asym_fall", "stride_asym_impulse"])
    return pd.DataFrame(records).set_index("step_id")


def build_extended_features(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice de features enrichies par pas.

    Nouvelles features (en plus des 7 originales) :
      rise_time_s, fall_time_s, rise_fall_ratio, impulse_front_ratio
        → calculées sur le signal filtré 10 Hz (robustesse au bruit).
      peak_force_subj_norm
        → peak divisé par la médiane de peak du sujet (invariance à la corpulence).
      stride_asym_rise, stride_asym_fall, stride_asym_impulse
        → asymétries intra-foulée des nouvelles features.
    """
    print("  [Features enrichies] Chargement et filtrage des signaux...")
    sig_cache = _load_filtered_signals(steps_df)

    # Base features (originales)
    df = build_step_features(steps_df)

    # Nouvelles features de forme — une ligne par pas
    shape_records: list[dict] = []
    n_missing = 0

    for _, row in steps_df.iterrows():
        sf = str(row["source_file"])
        foot = str(row["foot"])
        start = int(row["sample_start"])
        end = int(row["sample_end"])

        if sf not in sig_cache or foot not in sig_cache[sf]:
            shape_records.append({"step_id": str(row["step_id"])})
            n_missing += 1
            continue

        window = sig_cache[sf][foot][start:end]
        feats = _step_shape_features(window)
        feats["step_id"] = str(row["step_id"])
        shape_records.append(feats)

    if n_missing:
        warnings.warn(f"  [Features enrichies] {n_missing} pas sans signal source.", RuntimeWarning)

    shape_df = pd.DataFrame(shape_records).set_index("step_id")
    df = df.set_index("step_id").join(shape_df).reset_index()

    # Normalisation par sujet : peak_force / médiane sujet
    subj_median_peak = df.groupby("subject_id")["peak_force"].transform("median")
    df["peak_force_subj_norm"] = df["peak_force"] / (subj_median_peak + _EPS)

    # Asymétries enrichies
    ext_asym = _compute_extended_asymmetries(df)
    df = df.set_index("step_id").join(ext_asym).reset_index()

    n_enriched = (df["rise_time_s"].notna()).sum()
    print(f"  [Features enrichies] {n_enriched}/{len(df)} pas avec features de forme valides.")

    return df


# ===========================================================================
# 2b. FEATURES PAR CAPTEUR INDIVIDUEL (P6)
# ===========================================================================


def _shannon_entropy(dist: np.ndarray) -> float:
    """Entropie de Shannon normalisée d'une distribution de pression."""
    d = np.abs(dist)
    s = d.sum()
    if s < _EPS:
        return 0.0
    p = d / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def build_sensor_features(steps_df: pd.DataFrame, ds: StepDataset) -> pd.DataFrame:
    """
    Construit 30 features par capteur individuel pour chaque pas.

    Features :
      - 16 mean_L1..mean_R8 : force moyenne par capteur sur le pas
      - 4 ratios spatiaux   : forefoot/heel ratio L/R, forefoot_asym, heel_dominance
      - 8 asymétries        : |mean(Li) - mean(Ri)| / (mean(Li) + mean(Ri) + eps)
      - 2 entropies         : Shannon sur la distribution de pression L/R

    Paramètres
    ----------
    steps_df : DataFrame avec step_id, source_file, sample_start, sample_end, foot
    ds       : StepDataset (pour charger les signaux via __getitem__)
    """
    print("  [Features capteurs] Extraction des 30 features par capteur...")
    records: list[dict] = []

    step_id_to_idx = {str(steps_df.iloc[i]["step_id"]): i for i in range(len(steps_df))}

    for row_idx in range(len(steps_df)):
        row = steps_df.iloc[row_idx]
        sid = str(row["step_id"])

        ds_idx = step_id_to_idx.get(sid)
        if ds_idx is None or ds_idx >= len(ds):
            records.append({"step_id": sid})
            continue

        sig, _, _ = ds[ds_idx]  # (T, 16)

        if sig.shape[0] == 0:
            records.append({"step_id": sid})
            continue

        means = sig.mean(axis=0)  # (16,)

        rec: dict = {"step_id": sid}

        # 16 means
        for j, col_name in enumerate(_SENSOR_COLS):
            rec[f"mean_{col_name}"] = float(means[j])

        means_L = means[:8]  # L1..L8
        means_R = means[8:]  # R1..R8

        # Forefoot = sensors 1,2,3 ; Heel = sensors 6,7,8
        ff_L = means_L[:3].mean()
        heel_L = means_L[5:8].mean()
        ff_R = means_R[:3].mean()
        heel_R = means_R[5:8].mean()

        rec["forefoot_ratio_L"] = float(ff_L / (heel_L + _EPS))
        rec["forefoot_ratio_R"] = float(ff_R / (heel_R + _EPS))
        rec["forefoot_asym"] = float(abs(ff_L - ff_R) / (ff_L + ff_R + _EPS))
        rec["heel_dominance"] = float(
            (means_L[6:8].mean() + means_R[6:8].mean())
            / (means_L[:2].mean() + means_R[:2].mean() + _EPS)
        )

        # 8 asymétries par paire de capteurs
        for j in range(8):
            mL = float(means_L[j])
            mR = float(means_R[j])
            rec[f"sensor_asym_{j+1}"] = abs(mL - mR) / (mL + mR + _EPS)

        # Entropies
        rec["entropy_L"] = _shannon_entropy(means_L)
        rec["entropy_R"] = _shannon_entropy(means_R)

        records.append(rec)

    sensor_df = pd.DataFrame(records)
    n_valid = sensor_df[_SENSOR_MEAN_NAMES[0]].notna().sum()
    print(f"  [Features capteurs] {n_valid}/{len(sensor_df)} pas avec features capteurs valides.")
    print(f"  {len(_SENSOR_FEATURE_NAMES)} features capteurs disponibles.")
    return sensor_df


# ===========================================================================
# 3. OPTIMISATION DU SEUIL DE DÉCISION
# ===========================================================================


def optimize_threshold(
    subj_y: np.ndarray,
    subj_prob: np.ndarray,
) -> dict:
    """
    Balaye les seuils [0.05, 0.95] et retourne le seuil optimal selon
    le critère de Youden (J = sensibilité + spécificité − 1).

    Retourne un dict avec :
      threshold_youden : seuil J-optimal
      threshold_f1     : seuil F1-optimal
      threshold_acc    : seuil accuracy-optimal
      metrics_at_youden: dict des métriques au seuil Youden
      curve_data       : dict {thresholds, accuracy, f1, youden} pour tracé
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    acc_vals, f1_vals, youden_vals = [], [], []

    for t in thresholds:
        y_pred = (subj_prob >= t).astype(int)
        acc_vals.append(accuracy_score(subj_y, y_pred))
        f1_vals.append(f1_score(subj_y, y_pred, zero_division=0))

        # Youden : sensibilité (TPR) + spécificité (TNR) - 1
        tp = np.sum((y_pred == 1) & (subj_y == 1))
        fn = np.sum((y_pred == 0) & (subj_y == 1))
        tn = np.sum((y_pred == 0) & (subj_y == 0))
        fp = np.sum((y_pred == 1) & (subj_y == 0))
        tpr = tp / (tp + fn + _EPS)
        tnr = tn / (tn + fp + _EPS)
        youden_vals.append(tpr + tnr - 1.0)

    acc_vals = np.array(acc_vals)
    f1_vals = np.array(f1_vals)
    youden_vals = np.array(youden_vals)

    t_youden = float(thresholds[np.argmax(youden_vals)])
    t_f1     = float(thresholds[np.argmax(f1_vals)])
    t_acc    = float(thresholds[np.argmax(acc_vals)])

    y_pred_best = (subj_prob >= t_youden).astype(int)
    try:
        auc = float(roc_auc_score(subj_y, subj_prob))
    except ValueError:
        auc = float("nan")

    return {
        "threshold_youden": t_youden,
        "threshold_f1":     t_f1,
        "threshold_acc":    t_acc,
        "metrics_at_youden": {
            "accuracy":     float(accuracy_score(subj_y, y_pred_best)),
            "balanced_acc": float(balanced_accuracy_score(subj_y, y_pred_best)),
            "f1":           float(f1_score(subj_y, y_pred_best, zero_division=0)),
            "roc_auc":      auc,
            "youden":       float(youden_vals[np.argmax(youden_vals)]),
        },
        "curve_data": {
            "thresholds": thresholds.tolist(),
            "accuracy":   acc_vals.tolist(),
            "f1":         f1_vals.tolist(),
            "youden":     youden_vals.tolist(),
        },
    }


def _oof_subject_preds(preds_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agrège les prédictions pas-level OOF → niveau sujet.
    Chaque sujet apparaît dans exactement un fold test (StratifiedGroupKFold).
    Retourne (subject_ids, y_true, mean_proba).
    """
    agg = (
        preds_df
        .groupby("subject_id")
        .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
        .reset_index()
        .sort_values("subject_id")
    )
    return (
        agg["subject_id"].values,
        agg["y_true"].values.astype(int),
        agg["y_prob"].values.astype(float),
    )


# ===========================================================================
# 4. STACKING LÉGER
# ===========================================================================


def run_stacking_cv(
    model_subj_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Stacking niveau sujet : LogReg entraînée sur les meta-features OOF.

    model_subj_preds : {model_name: (subject_ids, y_true, mean_proba)}
    Tous les modèles doivent avoir le même ensemble de sujets.

    Retour : (metrics_dict, y_true_subj, stacker_proba)
    Évaluation : LOOCV sur sujets (leave-one-subject-out).
    Pas de fuite : les probas en entrée du stacker sont déjà OOF.
    """
    model_names = list(model_subj_preds.keys())

    # Aligner les sujets sur l'intersection
    ref_subjects = None
    for name, (sids, _, _) in model_subj_preds.items():
        s_set = set(sids)
        ref_subjects = s_set if ref_subjects is None else ref_subjects & s_set

    ref_subjects_sorted = sorted(ref_subjects)
    n = len(ref_subjects_sorted)

    # Construire la matrice (n_subjects, n_models)
    subj_idx = {s: i for i, s in enumerate(ref_subjects_sorted)}
    X_meta = np.zeros((n, len(model_names)), dtype=np.float64)
    y_subj = np.zeros(n, dtype=int)

    for m_idx, name in enumerate(model_names):
        sids, y_true, y_prob = model_subj_preds[name]
        for sid, yt, yp in zip(sids, y_true, y_prob):
            if sid in subj_idx:
                i = subj_idx[sid]
                X_meta[i, m_idx] = yp
                y_subj[i] = int(yt)

    # LOOCV sur le stacker
    loo = LeaveOneOut()
    stacker_proto = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    scaler_proto  = StandardScaler()
    oof_proba = np.zeros(n, dtype=float)

    for tr, te in loo.split(X_meta):
        X_tr = scaler_proto.fit_transform(X_meta[tr])
        X_te = scaler_proto.transform(X_meta[te])
        stacker_proto.fit(X_tr, y_subj[tr])
        p = stacker_proto.predict_proba(X_te)
        # Récupère la proba de la classe 1 (PD)
        classes = stacker_proto.classes_
        pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 0
        oof_proba[te] = p[:, pos_idx]

    oof_pred = (oof_proba >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_subj, oof_proba))
    except ValueError:
        auc = float("nan")

    metrics = {
        "model": "Stacking_RF_LogReg",
        "level": "subject",
        "Accuracy":     float(accuracy_score(y_subj, oof_pred)),
        "Balanced_Acc": float(balanced_accuracy_score(y_subj, oof_pred)),
        "F1-Score":     float(f1_score(y_subj, oof_pred, zero_division=0)),
        "ROC-AUC":      auc,
        "n_subjects":   n,
        "models_used":  model_names,
    }
    return metrics, y_subj, oof_proba


# ===========================================================================
# 5. FIGURES
# ===========================================================================


def plot_signal_filter_comparison(steps_df: pd.DataFrame, n_examples: int = 4) -> None:
    """
    Compare le signal brut et le signal filtré (Butterworth 10 Hz) sur
    quelques pas représentatifs (2 PD + 2 CO).
    Vérifie visuellement que la structure des pas est préservée.
    """
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Sélectionner 2 PD + 2 CO avec qualité "ok"
    ok_steps = steps_df[steps_df["quality_flag"] == "ok"].copy()
    pd_sample = ok_steps[ok_steps["group"] == "PD"].sample(
        min(2, (ok_steps["group"] == "PD").sum()), random_state=RANDOM_STATE
    )
    co_sample = ok_steps[ok_steps["group"] == "CO"].sample(
        min(2, (ok_steps["group"] == "CO").sum()), random_state=RANDOM_STATE
    )
    examples = pd.concat([pd_sample, co_sample]).reset_index(drop=True)

    n_axes = len(examples)
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 4), sharey=False)
    if n_axes == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, examples.iterrows()):
        sf = str(row["source_file"])
        foot = str(row["foot"])
        start = int(row["sample_start"])
        end = int(row["sample_end"])

        try:
            sig = load_signal_file(REPO_ROOT / Path(sf))
            col = f"total_{foot}"
            raw = sig[col].values[start:end].astype(np.float32)
        except Exception:
            continue

        filtered = _butter_filter(raw)
        t = np.arange(len(raw)) / _FS * 1000  # ms

        ax.plot(t, raw,      color="steelblue", alpha=0.5, linewidth=1.0, label="Brut")
        ax.plot(t, filtered, color="crimson",   alpha=0.9, linewidth=1.5, label="Filtré 10 Hz")
        ax.set_title(f"{row['group']} — {row['subject_id']}\nPied {foot}, {row['duration_s']:.2f} s",
                     fontsize=9)
        ax.set_xlabel("Temps (ms)")
        ax.set_ylabel("Force totale")
        ax.legend(fontsize=8)

    fig.suptitle("Comparaison signal brut vs filtré Butterworth 10 Hz\n"
                 "(structure des pas préservée — pics identiques)", fontsize=11)
    save_fig(fig, _FIG_DIR, "signal_filter_comparison")
    print(f"  -> {_FIG_DIR}/signal_filter_comparison.png")


def plot_threshold_curve(result: dict) -> None:
    """Courbes accuracy / F1 / Youden en fonction du seuil de décision."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    cd = result["curve_data"]
    t = np.array(cd["thresholds"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, cd["accuracy"], label="Accuracy",        color="steelblue",  linewidth=1.8)
    ax.plot(t, cd["f1"],       label="F1-Score",        color="seagreen",   linewidth=1.8)
    ax.plot(t, cd["youden"],   label="Youden (J)",      color="darkorange", linewidth=1.8)

    for key, color, ls, label in [
        ("threshold_youden", "darkorange", "--", f"Seuil Youden = {result['threshold_youden']:.2f}"),
        ("threshold_f1",     "seagreen",   ":",  f"Seuil F1-max = {result['threshold_f1']:.2f}"),
        ("threshold_acc",    "steelblue",  "-.", f"Seuil Acc-max = {result['threshold_acc']:.2f}"),
    ]:
        ax.axvline(result[key], color=color, linestyle=ls, linewidth=1.2, alpha=0.8, label=label)

    m = result["metrics_at_youden"]
    textstr = (
        f"Au seuil Youden ({result['threshold_youden']:.2f}) :\n"
        f"  Balanced Acc = {m['balanced_acc']:.3f}\n"
        f"  F1           = {m['f1']:.3f}\n"
        f"  ROC-AUC      = {m['roc_auc']:.3f}"
    )
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Score")
    ax.set_title("Optimisation du seuil de décision — niveau sujet (RF OOF)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    save_fig(fig, _FIG_DIR, "threshold_curve")
    print(f"  -> {_FIG_DIR}/threshold_curve.png")


def plot_performance_comparison(results_df: pd.DataFrame) -> None:
    """
    Barplot comparatif : baseline (features originales) vs enrichi + stacking.
    Un graphique par métrique, séparé niveau sujet.
    """
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]

    subj = results_df[results_df["level"] == "subject"].copy()
    if subj.empty:
        return

    palette = {
        "RF_baseline":     "#4878CF",
        "LogReg_baseline": "#6ACC65",
        "RF_enriched":     "#D65F5F",
        "LogReg_enriched": "#B47CC7",
        "Stacking_RF_LogReg": "#C4AD66",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        summary = (
            subj.groupby("model")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        colors = [palette.get(m, "#999999") for m in summary["model"]]
        bars = ax.bar(summary["model"], summary["mean"], color=colors,
                      yerr=summary["std"], capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        for bar, mean in zip(bars, summary["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
        # Ligne de référence cible
        ax.axhline(0.75, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Cible 0.75")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Comparaison baseline vs features enrichies + stacking — niveau sujet", fontsize=12)

    # Légende manuelle
    patches = [mpatches.Patch(color=c, label=m) for m, c in palette.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))

    save_fig(fig, _FIG_DIR, "performance_comparison")
    print(f"  -> {_FIG_DIR}/performance_comparison.png")


def plot_stacking_comparison(stacking_metrics: dict, baseline_subj: pd.DataFrame) -> None:
    """Barplot : stacking vs modèles individuels au niveau sujet."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]

    # Construire un DataFrame de comparaison
    rows = []
    for _, grp in baseline_subj.groupby("model"):
        for m in metrics:
            rows.append({
                "model":  grp["model"].iloc[0],
                "metric": m,
                "value":  grp[m].mean(),
            })

    for m in metrics:
        rows.append({
            "model":  "Stacking_RF_LogReg",
            "metric": m,
            "value":  stacking_metrics.get(m, np.nan),
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    model_order = df.groupby("model")["value"].mean().sort_values(ascending=False).index.tolist()
    sns.barplot(data=df, x="model", y="value", hue="metric",
                order=model_order, ax=ax, palette="Set2")
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Cible 0.75")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stacking vs modèles individuels — niveau sujet")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9, bbox_to_anchor=(1, 1))
    save_fig(fig, _FIG_DIR, "stacking_comparison")
    print(f"  -> {_FIG_DIR}/stacking_comparison.png")


# ===========================================================================
# 6. POINT D'ENTRÉE PRINCIPAL
# ===========================================================================


def run_improvements(session: str = "1", hybrid_preds_df: Optional[pd.DataFrame] = None) -> None:
    """
    Orchestration complète des 4 axes d'amélioration.

    1. Chargement des données (StepDataset).
    2. Filtrage du signal + features enrichies.
    3. CV StratifiedGroupKFold sur features originales (baseline) et enrichies.
    4. Optimisation du seuil de décision (Youden) sur OOF RF baseline.
    5. Stacking RF + LogReg via LOOCV sujet sur probas OOF.
    6. Génération des figures et exports CSV.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AMÉLIORATIONS PERFORMANCES STEP-LEVEL — GaitPDB")
    print("=" * 65)

    # ── 1. Chargement ──────────────────────────────────────────────────────
    print("\n[1/7] Chargement des pas (StepDataset)...")
    ds = StepDataset()
    steps_df = ds.df.copy()

    if session != "all":
        steps_df = steps_df[steps_df["session"].astype(str) == session].reset_index(drop=True)
        print(f"  Session {session} : {len(steps_df)} pas | {steps_df['subject_id'].nunique()} sujets")

    print(f"  Groupes : {dict(steps_df['group'].value_counts())}")

    # ── 2. Features originales (baseline) ──────────────────────────────────
    print("\n[2/7] Features originales (baseline)...")
    base_features_df = build_step_features(steps_df)
    X_base = base_features_df[_TAB_FEATURE_NAMES].values.astype(np.float32)
    y      = (steps_df["group"] == "PD").astype(int).values
    groups = steps_df["subject_id"].to_numpy(dtype=object)

    print(f"  {len(_TAB_FEATURE_NAMES)} features : {_TAB_FEATURE_NAMES}")

    # ── 3. CV baseline ─────────────────────────────────────────────────────
    print(f"\n[3/7] CV baseline ({_N_SPLITS}-fold StratifiedGroupKFold)...")
    baseline_results: list[dict] = []
    baseline_preds: dict[str, pd.DataFrame] = {}

    for name, clf in get_classifiers().items():
        print(f"  -> {name} (baseline)...")
        cv_res, preds_df = run_classifier_cv(name, clf, X_base, y, groups, steps_df)
        for r in cv_res:
            r["model"] = f"{name}_baseline"
        baseline_results.extend(cv_res)
        baseline_preds[name] = preds_df

    # ── 4. Features enrichies ──────────────────────────────────────────────
    print("\n[4/7] Construction des features enrichies...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        enrich_features_df = build_extended_features(steps_df)

    # Vérifier que les features enrichies sont disponibles
    avail_enrich = [f for f in _ENRICH_FEATURE_NAMES if f in enrich_features_df.columns]
    missing = set(_ENRICH_FEATURE_NAMES) - set(avail_enrich)
    if missing:
        warnings.warn(f"Features absentes (NaN → médiane) : {missing}", RuntimeWarning)

    X_enrich = enrich_features_df.reindex(columns=avail_enrich).values.astype(np.float32)
    print(f"  {len(avail_enrich)} features enrichies disponibles.")

    # Export features enrichies
    enrich_path = _OUT_DIR / "features_enriched.csv"
    enrich_features_df.to_csv(enrich_path, index=False)
    print(f"  -> {enrich_path}")

    # ── 4b. Features capteurs (P6) ────────────────────────────────────────
    print("\n[4b/9] Construction des features capteurs individuels...")
    ds_for_sensors = StepDataset(cache_signals=True)
    if session != "all":
        ds_for_sensors.df = steps_df.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sensor_features_df = build_sensor_features(steps_df, ds_for_sensors)

    # Fusionner enrichies + capteurs → 45 features
    enrich_indexed = enrich_features_df.set_index("step_id")
    sensor_indexed = sensor_features_df.set_index("step_id")
    all_tab_df = enrich_indexed.join(sensor_indexed, rsuffix="_sensor").reset_index()

    avail_all = [f for f in _ALL_TAB_FEATURE_NAMES if f in all_tab_df.columns]
    X_all = all_tab_df.reindex(columns=avail_all).values.astype(np.float32)
    print(f"  {len(avail_all)} features totales (enrichies + capteurs).")

    sensor_path = _OUT_DIR / "features_sensor.csv"
    sensor_features_df.to_csv(sensor_path, index=False)
    print(f"  -> {sensor_path}")

    # ── 5. CV enrichi ──────────────────────────────────────────────────────
    print(f"\n[5/9] CV features enrichies ({_N_SPLITS}-fold StratifiedGroupKFold)...")
    enriched_results: list[dict] = []
    enriched_preds: dict[str, pd.DataFrame] = {}

    for name, clf in get_classifiers().items():
        print(f"  -> {name} (enrichi)...")
        cv_res, preds_df = run_classifier_cv(name, clf, X_enrich, y, groups, steps_df)
        for r in cv_res:
            r["model"] = f"{name}_enriched"
        enriched_results.extend(cv_res)
        enriched_preds[name] = preds_df

    # ── 5b. CV features complètes (enrichies + capteurs) ───────────────────
    print(f"\n[5b/9] CV features complètes ({_N_SPLITS}-fold StratifiedGroupKFold)...")
    full_results: list[dict] = []
    full_preds: dict[str, pd.DataFrame] = {}

    for name, clf in get_classifiers().items():
        print(f"  -> {name} (full 45 features)...")
        cv_res, preds_df = run_classifier_cv(name, clf, X_all, y, groups, steps_df)
        for r in cv_res:
            r["model"] = f"{name}_full"
        full_results.extend(cv_res)
        full_preds[name] = preds_df

    # ── 6. Optimisation du seuil ────────────────────────────────────────────
    print("\n[6/9] Optimisation du seuil de décision (RF baseline OOF)...")
    rf_sids, rf_y, rf_prob = _oof_subject_preds(baseline_preds["RF"])
    threshold_result = optimize_threshold(rf_y, rf_prob)

    t_opt = threshold_result["threshold_youden"]
    m_opt = threshold_result["metrics_at_youden"]
    print(f"  Seuil Youden optimal = {t_opt:.3f}")
    print(f"  Balanced Acc @ seuil = {m_opt['balanced_acc']:.3f}")
    print(f"  F1-Score     @ seuil = {m_opt['f1']:.3f}")
    print(f"  ROC-AUC              = {m_opt['roc_auc']:.3f}")

    threshold_path = _OUT_DIR / "threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as fh:
        json.dump({
            "threshold_youden": t_opt,
            "threshold_f1":     threshold_result["threshold_f1"],
            "threshold_acc":    threshold_result["threshold_acc"],
            "metrics_at_youden": m_opt,
            "note": "Seuil calculé sur OOF RF baseline — session 01",
        }, fh, indent=2)
    print(f"  -> {threshold_path}")

    # ── 7. Stacking enrichi ──────────────────────────────────────────────
    print("\n[7/9] Stacking leger RF_enriched + LogReg_enriched -> LogReg sujet...")
    model_subj_preds = {}
    for name, preds_df in enriched_preds.items():
        sids, yt, yp = _oof_subject_preds(preds_df)
        model_subj_preds[name] = (sids, yt, yp)

    stacking_metrics, y_subj, stack_proba = run_stacking_cv(model_subj_preds)
    print(f"  Stacking Balanced Acc = {stacking_metrics['Balanced_Acc']:.3f}")
    print(f"  Stacking ROC-AUC      = {stacking_metrics['ROC-AUC']:.3f}")

    # ── 8. Stacking full (enrichi + capteurs) ──────────────────────────────
    print("\n[8/9] Stacking RF_full + LogReg_full -> LogReg sujet...")
    model_subj_preds_full = {}
    for name, preds_df in full_preds.items():
        sids, yt, yp = _oof_subject_preds(preds_df)
        model_subj_preds_full[name] = (sids, yt, yp)

    stacking_full_metrics, _, _ = run_stacking_cv(model_subj_preds_full)
    stacking_full_metrics["model"] = "Stacking_RF_LogReg_full"
    print(f"  Stacking full Balanced Acc = {stacking_full_metrics['Balanced_Acc']:.3f}")
    print(f"  Stacking full ROC-AUC      = {stacking_full_metrics['ROC-AUC']:.3f}")

    # ── 8b. Ensemble Final (HybridCNN + LogReg_full + RF_full) ─────────────
    ensemble_metrics = None
    if hybrid_preds_df is not None and not hybrid_preds_df.empty:
        print("\n[8b/9] Ensemble Final : HybridCNN + LogReg_full + RF_full -> LogReg sujet...")
        hybrid_sids, hybrid_yt, hybrid_yp = _oof_subject_preds(hybrid_preds_df)
        ensemble_preds = {
            "HybridCNN": (hybrid_sids, hybrid_yt, hybrid_yp),
        }
        for name, preds_df in full_preds.items():
            sids, yt, yp = _oof_subject_preds(preds_df)
            ensemble_preds[f"{name}_full"] = (sids, yt, yp)

        ensemble_metrics, _, _ = run_stacking_cv(ensemble_preds)
        ensemble_metrics["model"] = "Ensemble_Final"
        print(f"  Ensemble Final Balanced Acc = {ensemble_metrics['Balanced_Acc']:.3f}")
        print(f"  Ensemble Final ROC-AUC      = {ensemble_metrics['ROC-AUC']:.3f}")
    else:
        print("\n  [SKIP] Pas de prédictions HybridCNN — Ensemble Final ignoré.")

    # ── 9. Assemblage et exports ───────────────────────────────────────────
    all_results = pd.DataFrame(baseline_results + enriched_results + full_results)

    # Ajouter stacking comme ligne unique (fold=all)
    stacking_row = {
        "model":        "Stacking_RF_LogReg",
        "fold":         0,
        "level":        "subject",
        "Accuracy":     stacking_metrics["Accuracy"],
        "Balanced_Acc": stacking_metrics["Balanced_Acc"],
        "F1-Score":     stacking_metrics["F1-Score"],
        "ROC-AUC":      stacking_metrics["ROC-AUC"],
    }
    stacking_full_row = {
        "model":        "Stacking_RF_LogReg_full",
        "fold":         0,
        "level":        "subject",
        "Accuracy":     stacking_full_metrics["Accuracy"],
        "Balanced_Acc": stacking_full_metrics["Balanced_Acc"],
        "F1-Score":     stacking_full_metrics["F1-Score"],
        "ROC-AUC":      stacking_full_metrics["ROC-AUC"],
    }
    stacking_rows = [stacking_row, stacking_full_row]
    stacking_csv_rows = [stacking_metrics, stacking_full_metrics]

    if ensemble_metrics is not None:
        ensemble_row = {
            "model":        "Ensemble_Final",
            "fold":         0,
            "level":        "subject",
            "Accuracy":     ensemble_metrics["Accuracy"],
            "Balanced_Acc": ensemble_metrics["Balanced_Acc"],
            "F1-Score":     ensemble_metrics["F1-Score"],
            "ROC-AUC":      ensemble_metrics["ROC-AUC"],
        }
        stacking_rows.append(ensemble_row)
        stacking_csv_rows.append(ensemble_metrics)

    all_results = pd.concat(
        [all_results, pd.DataFrame(stacking_rows)], ignore_index=True
    )

    # CSV stacking
    stack_path = _OUT_DIR / "stacking_results.csv"
    pd.DataFrame(stacking_csv_rows).to_csv(stack_path, index=False)
    print(f"  -> {stack_path}")

    # CSV résumé complet
    summary_path = _OUT_DIR / "baseline_plus_improvements.csv"
    all_results.to_csv(summary_path, index=False)
    print(f"  -> {summary_path}")

    # ── Résumé console ─────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("RESUME - niveau sujet (Balanced_Acc moyen par modele)")
    print("-" * 65)
    subj_summary = (
        all_results[all_results["level"] == "subject"]
        .groupby("model")[["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]]
        .mean()
        .sort_values("Balanced_Acc", ascending=False)
    )
    print(subj_summary.round(4).to_string())

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGénération des figures...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_signal_filter_comparison(steps_df)
        plot_threshold_curve(threshold_result)
        plot_performance_comparison(all_results)
        plot_stacking_comparison(
            stacking_metrics,
            all_results[all_results["level"] == "subject"],
        )

    print("\n" + "=" * 65)
    print("  TERMINÉ — output/etude_du_pas/")
    print("=" * 65)
    return all_results


if __name__ == "__main__":
    run_improvements()
