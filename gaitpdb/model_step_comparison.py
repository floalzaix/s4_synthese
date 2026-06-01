"""
[ROLE]
Comparaison de modèles au niveau du pas (step-level) sur GaitPDB.

[RESPONSIBILITY]
- Construire une matrice de features par pas (durée, force, asymétrie intra-foulée).
- Benchmarker RF, KMeans, Fuzzy C-Means et CNN 1D sur classification PD/CO par pas.
- Évaluer les modèles via GroupKFold (split par sujet, sans fuite de données).
- Identifier les pas "malades" (CO avec prédiction PD) et les pas fortement asymétriques.
- Générer graphiques et CSV dans output/etude_du_pas/.

[NOTE MÉTHODOLOGIQUE]
La classification par pas n'est pas équivalente à la classification par sujet.
Un sujet PD peut avoir des "bons" pas ; un sujet CO peut avoir des pas PD-like.
Les métriques sujet-agrégées (majority vote) sont le vrai indicateur de performance clinique.

[OUTPUTS]
- output/etude_du_pas/model_step_comparison_cv.csv
- output/etude_du_pas/phase_step_baseline_summary.csv
- output/etude_du_pas/focus_malades_premiers_pas.csv
- output/etude_du_pas/figures/model_step_comparison/*.png

[DEPENDENCIES]
- scikit-learn
- skfuzzy (optionnel — FCM)
- torch (optionnel — CNN 1D)
- gaitpdb.step_dataset, gaitpdb.config, gaitpdb.viz.utils
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, filtfilt

from gaitpdb.config import OUTPUT_DIR, RANDOM_STATE
from gaitpdb.step_dataset import StepDataset, SubjectSplitter
from gaitpdb.viz.utils import save_fig, setup_style

try:
    import skfuzzy as fuzz
    _HAS_SKFUZZY = True
except ImportError:
    _HAS_SKFUZZY = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

setup_style()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_STEP_OUT_DIR: Path = OUTPUT_DIR / "etude_du_pas"
_STEP_FIG_DIR: Path = _STEP_OUT_DIR / "figures" / "model_step_comparison"
_EPS: float = 1e-9
_FS: float = 100.0
_FILTER_CUTOFF: float = 10.0
_FILTER_ORDER: int = 4
_N_SPLITS: int = 5

# Padding CNN : 1.5 s @ 100 Hz — couvre le 98e percentile de durée des pas GaitPDB.
# Au-delà, les pas sont tronqués (représentent < 0.2% du dataset).
_CNN_PAD_LEN: int = 150
_CNN_EPOCHS: int = 20
_CNN_BATCH: int = 128

# FCM
_FCM_M: float = 2.0
_FCM_ERROR: float = 1e-5
_FCM_MAXITER: int = 500

# Seuils d'asymétrie pour la table "focus" :
# > 15% asymétrie de force pic : seuil clinique de marche pathologique (Robinson et al. 1987).
# > 10% asymétrie de durée : référence NeuroCom/Ga dataset, valeur observée 9e décile CO.
_ASYM_PEAK_THRESHOLD: float = 0.15
_ASYM_DURATION_THRESHOLD: float = 0.10

# Features tabulaires utilisées par RF, KMeans, FCM.
_TAB_FEATURE_NAMES: list[str] = [
    "duration_s",
    "peak_force",
    "auc_norm",
    "foot_L",
    "stride_asym_peak",
    "stride_asym_duration",
    "stride_asym_auc",
]


# ---------------------------------------------------------------------------
# Construction des features pas-level
# ---------------------------------------------------------------------------


def _compute_stride_asymmetries(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les asymétries intra-foulée (L vs R) pour les pas appariés.

    Stratégie : pour chaque (subject_id, session, stride_id), on joint les pas
    L et R via stride_id. Les pas non appariés (stride_id = NaN) reçoivent NaN.
    L'asymétrie est définie comme |L - R| / (L + R + eps) — symétrique, bornée [0, 1].

    Retourne un DataFrame indexé sur step_id avec les colonnes :
      stride_asym_peak, stride_asym_duration, stride_asym_auc.
    """
    paired = steps_df[steps_df["stride_id"].notna()].copy()
    paired["stride_id_int"] = paired["stride_id"].astype(int)

    asym_records: list[dict] = []
    key = ["subject_id", "session", "stride_id_int"]

    for _, grp in paired.groupby(key):
        if set(grp["foot"].unique()) != {"L", "R"}:
            continue
        l = grp[grp["foot"] == "L"].iloc[0]
        r = grp[grp["foot"] == "R"].iloc[0]

        def _asym(a: float, b: float) -> float:
            return abs(a - b) / (a + b + _EPS)

        a_peak = _asym(l["peak_force"], r["peak_force"])
        a_dur  = _asym(l["duration_s"],  r["duration_s"])
        a_auc  = _asym(l["auc_force"],   r["auc_force"])

        for step_id in (l["step_id"], r["step_id"]):
            asym_records.append({
                "step_id": step_id,
                "stride_asym_peak":     a_peak,
                "stride_asym_duration": a_dur,
                "stride_asym_auc":      a_auc,
            })

    if not asym_records:
        return pd.DataFrame(columns=["stride_asym_peak", "stride_asym_duration", "stride_asym_auc"])
    asym_df = pd.DataFrame(asym_records).set_index("step_id")
    return asym_df


def build_step_features(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice de features tabulaires par pas.

    Features :
      - duration_s       : durée de la phase d'appui
      - peak_force       : force maximale durant l'appui
      - auc_norm         : auc_force / duration_s = force moyenne (sans biais de durée)
      - foot_L           : 1 si pied gauche, 0 si droit
      - stride_asym_*    : asymétries intra-foulée (NaN pour les pas non appariés)

    Les NaN (pas non appariés, ~5% des pas) sont imputés par la médiane dans les
    pipelines RF et clustering — imputation par colonne, après split train/test.
    """
    df = steps_df.copy()
    df["auc_norm"] = df["auc_force"] / (df["duration_s"] + _EPS)
    df["foot_L"]   = (df["foot"] == "L").astype(float)

    asym_df = _compute_stride_asymmetries(df)
    df = df.set_index("step_id").join(asym_df[["stride_asym_peak",
                                                "stride_asym_duration",
                                                "stride_asym_auc"]])
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Utilitaires CV
# ---------------------------------------------------------------------------


def _cv_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fold: int,
    model: str,
    level: str,
) -> dict:
    """Calcule les métriques standard pour un fold donné."""
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "model": model,
        "fold": fold,
        "level": level,
        "Accuracy":     float(accuracy_score(y_true, y_pred)),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "F1-Score":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ROC-AUC":      auc,
    }


def _optimize_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trouve le seuil maximisant l'indice de Youden (J = TPR + TNR - 1)."""
    best_j, best_t = -1.0, 0.5
    for t in np.arange(0.10, 0.90, 0.005):
        preds = (y_prob >= t).astype(int)
        ba = balanced_accuracy_score(y_true, preds)
        j = 2 * ba - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def _aggregate_by_subject(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agrège les prédictions pas-level vers le niveau sujet.

    Agrégation :
      - y_prob : moyenne des probabilités PD par sujet
      - y_pred : basé sur y_prob >= threshold (seuil optimisé par Youden sur train)
      - y_true : label vrai du sujet (constant par sujet, pris en moyenne arrondie)
    """
    subj_y, subj_pred, subj_prob = [], [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        subj_y.append(round(float(y_true[mask].mean())))
        mean_prob = float(y_prob[mask].mean())
        subj_prob.append(mean_prob)
        subj_pred.append(int(mean_prob >= threshold))
    return (
        np.array(subj_y, dtype=int),
        np.array(subj_pred, dtype=int),
        np.array(subj_prob, dtype=float),
    )


def _align_cluster_to_pd(u: np.ndarray, y_train: np.ndarray) -> int:
    """
    Retourne l'indice du cluster FCM/KMeans qui correspond à PD.
    Le cluster dont les membres ont la proportion de PD la plus élevée = cluster PD.
    u : (c, n_train) — membership matrix (FCM) ou membership one-hot (KMeans).
    """
    hard_labels = u.argmax(axis=0)  # (n_train,)
    pd_fracs = [
        y_train[hard_labels == k].mean() if (hard_labels == k).any() else 0.0
        for k in range(u.shape[0])
    ]
    return int(np.argmax(pd_fracs))


# ---------------------------------------------------------------------------
# Helper proba sécurisé
# ---------------------------------------------------------------------------


def _predict_proba_safe(clf, X_te: np.ndarray) -> np.ndarray:
    """
    Retourne les probabilités pour la classe positive (1 = PD).
    Gère les cas dégénérés : fold entraîné sur une seule classe,
    ou classifieur sans predict_proba (decision_function en fallback).
    """
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_te)
        # Pour un Pipeline, les classes sont sur le dernier estimateur
        final = clf[-1] if hasattr(clf, "__getitem__") else clf
        classes = np.asarray(getattr(final, "classes_", [0, 1]))
        pos_idx = np.where(classes == 1)[0]
        if len(pos_idx) > 0 and proba.shape[1] > pos_idx[0]:
            return proba[:, pos_idx[0]]
        # Une seule classe vue à l'entraînement → probabilité PD = 0
        return np.zeros(len(X_te), dtype=float)
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X_te).astype(float)
    return clf.predict(X_te).astype(float)


# ---------------------------------------------------------------------------
# Modèles supervisés (RF, LogReg)
# ---------------------------------------------------------------------------


def get_classifiers() -> dict:
    """Retourne les classifieurs supervisés tabulaires."""
    return {
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                           class_weight="balanced", n_jobs=-1)),
        ]),
        "LogReg": Pipeline([
            ("imp",  SimpleImputer(strategy="median")),
            ("sc",   StandardScaler()),
            ("clf",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                        class_weight="balanced")),
        ]),
    }


def run_classifier_cv(
    name: str,
    clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    GroupKFold CV pour un classifieur supervisé.
    Retourne (cv_results, step_predictions_df).
    step_predictions_df : step_id, subject_id, y_true, y_pred, y_prob, fold.
    """
    gkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: list[dict] = []
    pred_records: list[dict] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        y_prob = _predict_proba_safe(clf, X[te])

        # Métriques pas-level
        cv_results.append(_cv_metrics(y[te], y_pred, y_prob, fold, name, "step"))

        # Seuil Youden optimisé sur les prédictions train agrégées par sujet
        tr_subj_true, _, tr_subj_prob = _aggregate_by_subject(
            groups[tr], y[tr], clf.predict(X[tr]),
            _predict_proba_safe(clf, X[tr]),
        )
        opt_threshold = _optimize_threshold_youden(tr_subj_true, tr_subj_prob)

        # Métriques sujet-agrégées avec seuil optimisé
        subj_true, subj_pred, subj_prob = _aggregate_by_subject(
            groups[te], y[te], y_pred, y_prob, threshold=opt_threshold,
        )
        cv_results.append(_cv_metrics(subj_true, subj_pred, subj_prob, fold, name, "subject"))

        for i, idx in enumerate(te):
            pred_records.append({
                "step_id":    steps_df.iloc[idx]["step_id"],
                "subject_id": groups[idx],
                "y_true":     int(y[idx]),
                "y_pred":     int(y_pred[i]),
                "y_prob":     float(y_prob[i]),
                "fold":       fold,
            })

    return cv_results, pd.DataFrame(pred_records)


# ---------------------------------------------------------------------------
# KMeans CV
# ---------------------------------------------------------------------------


def run_kmeans_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    GroupKFold CV pour KMeans (K=2).
    Les labels 0/1 sont alignés sur PD/CO par proportion de PD dans le cluster train.
    """
    gkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    cv_results: list[dict] = []
    pred_records: list[dict] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr = scaler.fit_transform(imputer.fit_transform(X[tr]))
        X_te = scaler.transform(imputer.transform(X[te]))

        km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_tr)

        # Membership one-hot pour l'alignement
        labels_tr = km.labels_
        u_tr = np.zeros((2, len(labels_tr)))
        for k in range(2):
            u_tr[k, labels_tr == k] = 1.0
        pd_cluster = _align_cluster_to_pd(u_tr, y[tr])

        labels_te = km.predict(X_te)
        # Probabilité = distance normalisée au centroïde PD (plus proche = probabilité plus haute)
        dists = km.transform(X_te)  # (n_te, 2) — distance à chaque centroïde
        d_pd = dists[:, pd_cluster]
        d_other = dists[:, 1 - pd_cluster]
        # Probabilité PD : proportion de proximité — inverse de la distance normalisée
        y_prob = d_other / (d_pd + d_other + _EPS)
        y_pred = (labels_te == pd_cluster).astype(int)

        cv_results.append(_cv_metrics(y[te], y_pred, y_prob, fold, "KMeans", "step"))

        # Seuil Youden sur train
        labels_tr_pred = (km.labels_ == pd_cluster).astype(int)
        dists_tr = km.transform(X_tr)
        tr_prob = dists_tr[:, 1 - pd_cluster] / (dists_tr[:, pd_cluster] + dists_tr[:, 1 - pd_cluster] + _EPS)
        tr_subj_true, _, tr_subj_prob = _aggregate_by_subject(
            groups[tr], y[tr], labels_tr_pred, tr_prob,
        )
        opt_threshold = _optimize_threshold_youden(tr_subj_true, tr_subj_prob)

        subj_true, subj_pred, subj_prob = _aggregate_by_subject(
            groups[te], y[te], y_pred, y_prob, threshold=opt_threshold,
        )
        cv_results.append(_cv_metrics(subj_true, subj_pred, subj_prob, fold, "KMeans", "subject"))

        for i, idx in enumerate(te):
            pred_records.append({
                "step_id":    steps_df.iloc[idx]["step_id"],
                "subject_id": groups[idx],
                "y_true":     int(y[idx]),
                "y_pred":     int(y_pred[i]),
                "y_prob":     float(y_prob[i]),
                "fold":       fold,
            })

    return cv_results, pd.DataFrame(pred_records)


# ---------------------------------------------------------------------------
# Fuzzy C-Means CV (optionnel)
# ---------------------------------------------------------------------------


def run_fcm_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    GroupKFold CV pour Fuzzy C-Means (C=2).
    Nécessite skfuzzy. La probabilité PD = degré d'appartenance au cluster PD.
    """
    if not _HAS_SKFUZZY:
        print("  [FCM] skfuzzy non disponible — étape ignorée.")
        return [], pd.DataFrame()

    gkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    cv_results: list[dict] = []
    pred_records: list[dict] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr = scaler.fit_transform(imputer.fit_transform(X[tr]))
        X_te = scaler.transform(imputer.transform(X[te]))

        # skfuzzy attend (n_features, n_samples)
        cntr, u_tr, *_ = fuzz.cmeans(
            X_tr.T, c=2, m=_FCM_M, error=_FCM_ERROR,
            maxiter=_FCM_MAXITER, seed=RANDOM_STATE,
        )
        u_te, *_ = fuzz.cmeans_predict(
            X_te.T, cntr, m=_FCM_M, error=_FCM_ERROR, maxiter=_FCM_MAXITER,
        )

        pd_cluster = _align_cluster_to_pd(u_tr, y[tr])
        y_prob = u_te[pd_cluster].astype(float)
        y_pred = (y_prob >= 0.5).astype(int)

        cv_results.append(_cv_metrics(y[te], y_pred, y_prob, fold, "FCM", "step"))

        # Seuil Youden sur train
        tr_prob_fcm = u_tr[pd_cluster].astype(float)
        tr_pred_fcm = (tr_prob_fcm >= 0.5).astype(int)
        tr_subj_true, _, tr_subj_prob = _aggregate_by_subject(
            groups[tr], y[tr], tr_pred_fcm, tr_prob_fcm,
        )
        opt_threshold = _optimize_threshold_youden(tr_subj_true, tr_subj_prob)

        subj_true, subj_pred, subj_prob = _aggregate_by_subject(
            groups[te], y[te], y_pred, y_prob, threshold=opt_threshold,
        )
        cv_results.append(_cv_metrics(subj_true, subj_pred, subj_prob, fold, "FCM", "subject"))

        for i, idx in enumerate(te):
            pred_records.append({
                "step_id":    steps_df.iloc[idx]["step_id"],
                "subject_id": groups[idx],
                "y_true":     int(y[idx]),
                "y_pred":     int(y_pred[i]),
                "y_prob":     float(y_prob[i]),
                "fold":       fold,
            })

    return cv_results, pd.DataFrame(pred_records)


# ---------------------------------------------------------------------------
# CNN 1D PyTorch (optionnel)
# ---------------------------------------------------------------------------


def _build_padded_signals(ds: StepDataset) -> np.ndarray:
    """
    Pré-charge tous les pas en mémoire, filtrés (Butterworth 10 Hz) et paddés à _CNN_PAD_LEN.
    Retourne un array float32 de shape (N, 16, _CNN_PAD_LEN).
    Les pas plus longs que _CNN_PAD_LEN sont tronqués (< 0.2% du dataset).
    """
    nyq = _FS / 2.0
    b, a = butter(_FILTER_ORDER, _FILTER_CUTOFF / nyq, btype="low")

    print(f"  [CNN] Chargement + filtrage des signaux ({len(ds)} pas x 16 capteurs x {_CNN_PAD_LEN} samples)...")
    X = np.zeros((len(ds), 16, _CNN_PAD_LEN), dtype=np.float32)
    for i in range(len(ds)):
        sig, _, _ = ds[i]          # (T, 16)
        t = min(sig.shape[0], _CNN_PAD_LEN)
        for ch in range(16):
            ch_data = sig[:t, ch].copy()
            n = len(ch_data)
            if n >= 4:
                pl = min(3 * max(len(a), len(b)) - 1, n - 1)
                ch_data = filtfilt(b, a, ch_data, padlen=pl).astype(np.float32)
            X[i, ch, :t] = ch_data
    return X


if _HAS_TORCH:
    class _StepCNN(nn.Module):
        """
        CNN 1D léger pour classification binaire PD/CO sur séquences de force.
        Architecture : 3 blocs Conv-BN-ReLU-Pool, puis GlobalAvgPool + Linear.
        Entrée : (batch, 16, T) — T = _CNN_PAD_LEN.
        Sortie : (batch, 2) — logits avant softmax.
        """
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool1d(2),                          # -> 75
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool1d(2),                          # -> 37
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(1),                  # -> (batch, 64, 1)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x).squeeze(-1)  # (batch, 64)
            return self.classifier(x)


if _HAS_TORCH:
    class _AugmentedDataset(TensorDataset):
        """
        Wrapper appliquant des augmentations on-the-fly aux signaux CNN.
        Utilisé uniquement pendant l'entraînement.

        Augmentations (p=0.5 chacune) :
          - Amplitude scaling : *U[0.9, 1.1]
          - Bruit gaussien : +N(0, 0.02*std_channel) par canal
          - Time shift : décalage ±5 samples avec zéro-padding
        """
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            if not self.training:
                return x, y
            x = x.clone()

            # Amplitude scaling
            if torch.rand(1).item() > 0.5:
                scale = 0.9 + 0.2 * torch.rand(1).item()
                x *= scale

            # Bruit gaussien
            if torch.rand(1).item() > 0.5:
                for ch in range(x.shape[0]):
                    std = x[ch].std().item()
                    if std > 1e-9:
                        x[ch] += torch.randn_like(x[ch]) * (0.02 * std)

            # Time shift
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-5, 6, (1,)).item()
                if shift != 0:
                    x_new = torch.zeros_like(x)
                    T = x.shape[1]
                    if shift > 0:
                        x_new[:, shift:] = x[:, :T - shift]
                    else:
                        x_new[:, :T + shift] = x[:, -shift:]
                    x = x_new

            return x, y

        training: bool = True


    class _HybridCNN(nn.Module):
        """
        CNN hybride : branche signal (convolutions) + branche tabulaire (MLP).
        Fusionne les deux embeddings avant la classification finale.
        """
        def __init__(self, n_tab_features: int) -> None:
            super().__init__()
            self.signal_branch = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(1),
            )
            self.tab_branch = nn.Sequential(
                nn.Linear(n_tab_features, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 + 32, 2),
            )

        def forward(self, x_signal: "torch.Tensor", x_tab: "torch.Tensor") -> "torch.Tensor":
            sig_emb = self.signal_branch(x_signal).squeeze(-1)  # (batch, 64)
            tab_emb = self.tab_branch(x_tab)                    # (batch, 32)
            fused = torch.cat([sig_emb, tab_emb], dim=1)        # (batch, 96)
            return self.classifier(fused)


def _plot_training_curves(curves: dict[int, dict], model_name: str = "CNN1D") -> None:
    """Génère un graphique des courbes train/val loss par fold CNN."""
    n_folds = len(curves)
    fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 3.5), squeeze=False)
    for fold_idx, (fold, data) in enumerate(sorted(curves.items())):
        ax = axes[0, fold_idx]
        epochs = range(1, len(data["train_loss"]) + 1)
        ax.plot(epochs, data["train_loss"], label="Train", color="steelblue")
        ax.plot(epochs, data["val_loss"], label="Val", color="tomato")
        ax.axvline(data["best_epoch"], color="green", linestyle="--", alpha=0.7,
                   label=f"Best (ep {data['best_epoch']})")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    fig.suptitle(f"{model_name} — Courbes d'entraînement par fold", fontsize=12)
    fig.tight_layout()
    fname = f"{model_name.lower()}_training_curves"
    save_fig(fig, _STEP_FIG_DIR, fname)
    print(f"  -> {_STEP_FIG_DIR}/{fname}.png")


def run_cnn_cv(
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    StratifiedGroupKFold CV pour CNN 1D PyTorch avec early stopping.
    X_raw : (N, 16, T_pad) float32.
    """
    if not _HAS_TORCH:
        print("  [CNN] torch non disponible — étape ignorée.")
        return [], pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [CNN] Device : {device}  | {_CNN_EPOCHS} epochs / fold")

    gkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: list[dict] = []
    pred_records: list[dict] = []
    training_curves: dict[int, dict] = {}

    X_t = torch.from_numpy(X_raw)
    y_t = torch.from_numpy(y.astype(np.int64))

    for fold, (tr, te) in enumerate(gkf.split(X_raw, y, groups=groups), 1):
        print(f"    Fold {fold}/{_N_SPLITS}...")
        model = _StepCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_CNN_EPOCHS)

        n_co = int((y[tr] == 0).sum())
        n_pd = int((y[tr] == 1).sum())
        n_total = n_co + n_pd
        class_weights = torch.tensor(
            [n_total / (2.0 * n_co + 1e-9), n_total / (2.0 * n_pd + 1e-9)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_ds = TensorDataset(X_t[tr], y_t[tr])
        loader = DataLoader(train_ds, batch_size=_CNN_BATCH, shuffle=True)

        # Petit val set pour monitoring uniquement (pas d'early stopping)
        tr_subjects = np.unique(groups[tr])
        rng = np.random.RandomState(RANDOM_STATE + fold)
        rng.shuffle(tr_subjects)
        n_val_subj = max(2, int(len(tr_subjects) * 0.15))
        val_subjects = set(tr_subjects[:n_val_subj])
        inner_val_mask = np.array([groups[i] in val_subjects for i in tr])
        inner_val_idx = tr[inner_val_mask]

        fold_train_losses: list[float] = []
        fold_val_losses: list[float] = []

        for epoch in range(_CNN_EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            fold_train_losses.append(epoch_loss / max(n_batches, 1))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_t[inner_val_idx].to(device)).cpu()
                val_loss = float(nn.CrossEntropyLoss()(val_logits, y_t[inner_val_idx]))
            fold_val_losses.append(val_loss)

        training_curves[fold] = {
            "train_loss": fold_train_losses,
            "val_loss": fold_val_losses,
            "best_epoch": _CNN_EPOCHS,
        }

        model.eval()
        with torch.no_grad():
            logits = model(X_t[te].to(device)).cpu()
            probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds  = logits.argmax(dim=1).numpy()

        cv_results.append(_cv_metrics(y[te], preds, probs, fold, "CNN1D", "step"))

        # Seuil Youden sur prédictions train (full train, pas inner)
        with torch.no_grad():
            tr_logits = model(X_t[tr].to(device)).cpu()
            tr_probs = torch.softmax(tr_logits, dim=1)[:, 1].numpy()
            tr_preds = tr_logits.argmax(dim=1).numpy()
        tr_subj_true, _, tr_subj_prob = _aggregate_by_subject(
            groups[tr], y[tr], tr_preds, tr_probs,
        )
        opt_threshold = _optimize_threshold_youden(tr_subj_true, tr_subj_prob)

        subj_true, subj_pred, subj_prob = _aggregate_by_subject(
            groups[te], y[te], preds, probs, threshold=opt_threshold,
        )
        cv_results.append(_cv_metrics(subj_true, subj_pred, subj_prob, fold, "CNN1D", "subject"))

        for i, idx in enumerate(te):
            pred_records.append({
                "step_id":    steps_df.iloc[idx]["step_id"],
                "subject_id": groups[idx],
                "y_true":     int(y[idx]),
                "y_pred":     int(preds[i]),
                "y_prob":     float(probs[i]),
                "fold":       fold,
            })

    # Export des courbes d'entraînement
    _plot_training_curves(training_curves)

    return cv_results, pd.DataFrame(pred_records)


# ---------------------------------------------------------------------------
# CNN Hybride (signal + features tabulaires)
# ---------------------------------------------------------------------------


def run_hybrid_cnn_cv(
    X_raw: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps_df: pd.DataFrame,
    use_augmentation: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """
    StratifiedGroupKFold CV pour CNN hybride (signal + features tabulaires).

    X_raw : (N, 16, T_pad) float32 — signaux filtrés paddés.
    X_tab : (N, n_features) float32 — features tabulaires (enrichies + capteurs).
    """
    if not _HAS_TORCH:
        print("  [HybridCNN] torch non disponible — étape ignorée.")
        return [], pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tab = X_tab.shape[1]
    print(f"  [HybridCNN] Device : {device} | {_CNN_EPOCHS} epochs/fold | "
          f"{n_tab} features tab | augmentation={'oui' if use_augmentation else 'non'}")

    gkf = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: list[dict] = []
    pred_records: list[dict] = []
    training_curves: dict[int, dict] = {}

    X_sig_t = torch.from_numpy(X_raw)
    y_t = torch.from_numpy(y.astype(np.int64))

    for fold, (tr, te) in enumerate(gkf.split(X_raw, y, groups=groups), 1):
        print(f"    Fold {fold}/{_N_SPLITS}...")

        # Imputation + normalisation des features tabulaires (fit sur train)
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tab_tr = scaler.fit_transform(imp.fit_transform(X_tab[tr]))
        X_tab_te = scaler.transform(imp.transform(X_tab[te]))
        X_tab_tr_t = torch.from_numpy(X_tab_tr.astype(np.float32))
        X_tab_te_t = torch.from_numpy(X_tab_te.astype(np.float32))

        model = _HybridCNN(n_tab).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_CNN_EPOCHS)

        n_co = int((y[tr] == 0).sum())
        n_pd = int((y[tr] == 1).sum())
        n_total = n_co + n_pd
        class_weights = torch.tensor(
            [n_total / (2.0 * n_co + 1e-9), n_total / (2.0 * n_pd + 1e-9)],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        if use_augmentation:
            train_ds = _AugmentedDataset(X_sig_t[tr], y_t[tr])
            train_ds.training = True
        else:
            train_ds = TensorDataset(X_sig_t[tr], y_t[tr])
        loader = DataLoader(train_ds, batch_size=_CNN_BATCH, shuffle=True)

        # Val set for monitoring
        tr_subjects = np.unique(groups[tr])
        rng = np.random.RandomState(RANDOM_STATE + fold)
        rng.shuffle(tr_subjects)
        n_val_subj = max(2, int(len(tr_subjects) * 0.15))
        val_subjects = set(tr_subjects[:n_val_subj])
        inner_val_mask = np.array([groups[i] in val_subjects for i in tr])
        inner_val_local = np.where(inner_val_mask)[0]

        fold_train_losses: list[float] = []
        fold_val_losses: list[float] = []

        for epoch in range(_CNN_EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx_start in range(0, len(tr), _CNN_BATCH):
                batch_end = min(batch_idx_start + _CNN_BATCH, len(tr))
                local_idx = list(range(batch_idx_start, batch_end))

                xb_sig = X_sig_t[tr[local_idx]].to(device)
                xb_tab = X_tab_tr_t[local_idx].to(device)
                yb = y_t[tr[local_idx]].to(device)

                if use_augmentation and torch.rand(1).item() > 0.5:
                    scale = 0.9 + 0.2 * torch.rand(1).item()
                    xb_sig = xb_sig * scale
                if use_augmentation and torch.rand(1).item() > 0.5:
                    noise_std = 0.02 * xb_sig.std(dim=-1, keepdim=True)
                    xb_sig = xb_sig + torch.randn_like(xb_sig) * noise_std

                optimizer.zero_grad()
                loss = criterion(model(xb_sig, xb_tab), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            fold_train_losses.append(epoch_loss / max(n_batches, 1))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(
                    X_sig_t[tr[inner_val_local]].to(device),
                    X_tab_tr_t[inner_val_local].to(device),
                ).cpu()
                val_loss = float(nn.CrossEntropyLoss()(val_logits, y_t[tr[inner_val_local]]))
            fold_val_losses.append(val_loss)

        training_curves[fold] = {
            "train_loss": fold_train_losses,
            "val_loss": fold_val_losses,
            "best_epoch": _CNN_EPOCHS,
        }

        model.eval()
        with torch.no_grad():
            logits = model(
                X_sig_t[te].to(device),
                X_tab_te_t.to(device),
            ).cpu()
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = logits.argmax(dim=1).numpy()

        cv_results.append(_cv_metrics(y[te], preds, probs, fold, "HybridCNN", "step"))

        # Youden threshold on train predictions
        with torch.no_grad():
            tr_logits = model(
                X_sig_t[tr].to(device),
                X_tab_tr_t.to(device),
            ).cpu()
            tr_probs = torch.softmax(tr_logits, dim=1)[:, 1].numpy()
            tr_preds = tr_logits.argmax(dim=1).numpy()
        tr_subj_true, _, tr_subj_prob = _aggregate_by_subject(
            groups[tr], y[tr], tr_preds, tr_probs,
        )
        opt_threshold = _optimize_threshold_youden(tr_subj_true, tr_subj_prob)

        subj_true, subj_pred, subj_prob = _aggregate_by_subject(
            groups[te], y[te], preds, probs, threshold=opt_threshold,
        )
        cv_results.append(_cv_metrics(subj_true, subj_pred, subj_prob, fold, "HybridCNN", "subject"))

        for i, idx in enumerate(te):
            pred_records.append({
                "step_id":    steps_df.iloc[idx]["step_id"],
                "subject_id": groups[idx],
                "y_true":     int(y[idx]),
                "y_pred":     int(preds[i]),
                "y_prob":     float(probs[i]),
                "fold":       fold,
            })

    # Export training curves
    _plot_training_curves(training_curves, model_name="HybridCNN")

    return cv_results, pd.DataFrame(pred_records)


# ---------------------------------------------------------------------------
# Table focus : pas malades et pas asymétriques
# ---------------------------------------------------------------------------


def build_focus_table(
    steps_df: pd.DataFrame,
    features_df: pd.DataFrame,
    rf_preds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit la table des pas "d'intérêt clinique" :

    1. "pd_mismatch"  : pas prédit PD par RF mais sujet appartient au groupe CO.
                        Signal : la biométrie de ce pas ressemble à un pas PD,
                        potentiellement indicateur précoce ou bruit de capteur.

    2. "high_asym"    : asymétrie de force pic > _ASYM_PEAK_THRESHOLD (0.15)
                        OU asymétrie de durée > _ASYM_DURATION_THRESHOLD (0.10).
                        Seuils empiriques : Robinson et al. 1987 (force), décile 90
                        du dataset CO GaitPDB (durée).

    3. "both"         : cumul des deux critères — cas les plus saillants.

    La table est générée en utilisant les prédictions du dernier fold disponible
    pour chaque pas. Si un pas apparaît dans plusieurs folds (cas impossible avec
    GroupKFold), le fold le plus récent est retenu.
    """
    # Dernière prédiction disponible par pas (1 seul fold par pas avec GroupKFold)
    last_preds = (
        rf_preds_df.sort_values("fold")
        .groupby("step_id")
        .last()
        .reset_index()
    )

    # Jointure avec features asymétrie
    feat_cols = ["step_id", "stride_asym_peak", "stride_asym_duration", "stride_asym_auc"]
    available = [c for c in feat_cols if c in features_df.columns]
    merged = last_preds.merge(
        features_df[available].drop_duplicates("step_id"),
        on="step_id", how="left",
    )

    # Jointure avec métadonnées du pas
    meta_cols = ["step_id", "group", "foot", "session", "walk_type",
                 "quality_flag", "duration_s", "peak_force", "stride_id"]
    meta_cols = [c for c in meta_cols if c in steps_df.columns]
    merged = merged.merge(
        steps_df[meta_cols].drop_duplicates("step_id"),
        on="step_id", how="left",
    )

    # Critère 1 : mismatch (CO sujet, mais RF prédit PD)
    crit_mismatch = (merged["group"] == "CO") & (merged["y_pred"] == 1)

    # Critère 2 : haute asymétrie
    crit_asym = (
        (merged["stride_asym_peak"].fillna(0) > _ASYM_PEAK_THRESHOLD) |
        (merged["stride_asym_duration"].fillna(0) > _ASYM_DURATION_THRESHOLD)
    )

    merged["anomaly_type"] = "none"
    merged.loc[crit_mismatch,             "anomaly_type"] = "pd_mismatch"
    merged.loc[crit_asym,                 "anomaly_type"] = "high_asym"
    merged.loc[crit_mismatch & crit_asym, "anomaly_type"] = "both"

    focus = merged[merged["anomaly_type"] != "none"].copy()

    col_order = [
        "step_id", "subject_id", "group", "foot", "session", "walk_type",
        "stride_id", "quality_flag", "duration_s", "peak_force",
        "stride_asym_peak", "stride_asym_duration", "stride_asym_auc",
        "y_pred", "y_prob", "fold", "anomaly_type",
    ]
    col_order = [c for c in col_order if c in focus.columns]
    return focus[col_order].sort_values(["anomaly_type", "y_prob"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Graphiques de comparaison
# ---------------------------------------------------------------------------


def _export_comparison_csv(df_cv: pd.DataFrame) -> None:
    """
    Exporte un tableau de comparaison mean ± std de tous les modèles dans _STEP_FIG_DIR.

    Structure :
      - Une ligne par (model, level).
      - Colonnes : {metric}_mean, {metric}_std pour Accuracy, Balanced_Acc, F1-Score, ROC-AUC.
      - Tri par Balanced_Acc_mean DESC pour chaque level.
    """
    metrics = ["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]
    agg = (
        df_cv.groupby(["model", "level"])[metrics]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index().sort_values(
        ["level", "Balanced_Acc_mean"], ascending=[True, False]
    )

    # Colonnes dans l'ordre : model, level, puis métriques intercalées (mean, std)
    ordered_cols = ["model", "level"]
    for m in metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    agg = agg[ordered_cols]

    out_path = _STEP_FIG_DIR / "model_comparison_table.csv"
    agg.to_csv(out_path, index=False)
    print(f"  -> CSV comparaison modèles : {out_path}")


def generate_comparison_plots(df_cv: pd.DataFrame) -> None:
    """
    Génère barplots + boxplots pour chaque métrique, séparés par niveau (step / subject).
    Exporte aussi model_comparison_table.csv dans le même dossier.
    """
    _STEP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _export_comparison_csv(df_cv)
    metrics = ["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]

    for level in ("step", "subject"):
        sub = df_cv[df_cv["level"] == level]
        if sub.empty:
            continue

        order = (
            sub.groupby("model")["Balanced_Acc"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        level_label = "pas-level" if level == "step" else "agrégé par sujet (majority vote)"

        for metric in metrics:
            # Barplot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=sub, x="model", y=metric, order=order,
                ax=ax, errorbar="sd", palette="viridis",
                hue="model", legend=False,
            )
            ax.set_title(f"Performance moyenne {level_label} — {metric} ({_N_SPLITS}-fold CV par sujet)")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            save_fig(fig, _STEP_FIG_DIR, f"step_{level}_barplot_{metric.lower().replace('-','_')}")

            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(
                data=sub, x="model", y=metric, order=order,
                ax=ax, palette="Set3",
                hue="model", legend=False,
            )
            ax.set_title(f"Distribution des scores {level_label} — {metric}")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            save_fig(fig, _STEP_FIG_DIR, f"step_{level}_boxplot_{metric.lower().replace('-','_')}")


def _plot_asymmetry_distribution(focus: pd.DataFrame) -> None:
    """
    Distribution de l'asymétrie de force par groupe (PD vs CO) pour les pas focalisés.
    """
    if focus.empty or "stride_asym_peak" not in focus.columns:
        return

    full_ds = StepDataset()
    full_feat = build_step_features(full_ds.df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Asymétrie peak
    for ax, col, thresh, title in zip(
        axes,
        ["stride_asym_peak", "stride_asym_duration"],
        [_ASYM_PEAK_THRESHOLD, _ASYM_DURATION_THRESHOLD],
        ["Asymétrie de force (peak)", "Asymétrie de durée"],
    ):
        plot_df = full_feat.dropna(subset=[col]).copy()
        sns.kdeplot(
            data=plot_df, x=col, hue="group",
            palette={"PD": "salmon", "CO": "skyblue"},
            fill=True, alpha=0.4, ax=ax,
        )
        ax.axvline(thresh, color="black", linestyle="--", linewidth=1.2,
                   label=f"Seuil focus ({thresh})")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.legend()

    fig.suptitle("Distribution des asymétries intra-foulée — tous les pas (PD vs CO)")
    save_fig(fig, _STEP_FIG_DIR, "step_asymmetry_distribution")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def run_step_comparison(session: str = "01") -> None:
    """
    Orchestration complète de la comparaison de modèles step-level.

    Étapes :
    1. Chargement des pas filtrés (StepDataset).
    2. Construction des features tabulaires.
    3. CV pour chaque modèle (RF, LogReg, KMeans, FCM, CNN si disponibles).
    4. Assemblage des résultats et export CSV.
    5. Identification des pas focalisés et export.
    6. Génération des graphiques.
    """
    _STEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _STEP_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MODEL STEP COMPARISON — GaitPDB pas-level")
    print("=" * 60)

    # ── 1. Chargement des données ────────────────────────────────────────
    print("\n[1/6] Chargement des pas (StepDataset, session filtrage: tous)...")
    ds = StepDataset(cache_signals=_HAS_TORCH)  # cache si CNN prévu
    steps_df = ds.df.copy()

    if session != "all":
        # Filtrage optionnel sur la session (session "01" = marche normale)
        steps_df = steps_df[steps_df["session"].astype(str) == session].copy()
        steps_df = steps_df.reset_index(drop=True)
        print(f"  Session filtrée : {session} -> {len(steps_df)} pas retenus")

    print(f"  {len(steps_df)} pas | {steps_df['subject_id'].nunique()} sujets")
    print(f"  Groupes : {dict(steps_df['group'].value_counts())}")
    print(f"  Flags   : {dict(steps_df['quality_flag'].value_counts())}")

    # ── 2. Features tabulaires ───────────────────────────────────────────
    print("\n[2/6] Construction des features tabulaires...")
    features_df = build_step_features(steps_df)

    n_valid = len(features_df)
    n_paired = features_df["stride_asym_peak"].notna().sum()
    print(f"  {n_valid} pas | {n_paired} appariés ({n_paired/n_valid:.1%}) | "
          f"{n_valid - n_paired} non appariés (NaN -> médiane dans les pipelines)")

    X = features_df[_TAB_FEATURE_NAMES].values.astype(np.float32)
    y = (steps_df["group"] == "PD").astype(int).values
    groups = steps_df["subject_id"].to_numpy(dtype=object)

    # ── 3. CV modèles ────────────────────────────────────────────────────
    print(f"\n[3/6] Cross-validation ({_N_SPLITS}-fold GroupKFold par sujet)...")

    all_cv_results: list[dict] = []
    rf_preds_df: Optional[pd.DataFrame] = None

    # Supervisés
    for name, clf in get_classifiers().items():
        print(f"  -> {name}...")
        cv_res, preds_df = run_classifier_cv(name, clf, X, y, groups, steps_df)
        all_cv_results.extend(cv_res)
        if name == "RF":
            rf_preds_df = preds_df

    # KMeans
    print("  -> KMeans...")
    cv_res, _ = run_kmeans_cv(X, y, groups, steps_df)
    all_cv_results.extend(cv_res)

    # FCM
    print("  -> Fuzzy C-Means...")
    cv_res, _ = run_fcm_cv(X, y, groups, steps_df)
    all_cv_results.extend(cv_res)

    # CNN 1D + Hybrid CNN
    print("  -> CNN 1D...")
    X_raw = None
    if _HAS_TORCH:
        ds_raw = StepDataset(cache_signals=True)
        if session != "all":
            ds_raw.df = steps_df.reset_index(drop=True)
        X_raw = _build_padded_signals(ds_raw)
        cv_res, _ = run_cnn_cv(X_raw, y, groups, steps_df)
        all_cv_results.extend(cv_res)
    else:
        print("  [CNN] torch non disponible — ignoré.")

    # Hybrid CNN (signal + features tabulaires enrichies + capteurs)
    if _HAS_TORCH and X_raw is not None:
        print("  -> Hybrid CNN (signal + tabulaire)...")
        from gaitpdb.step_improvements import (
            build_extended_features,
            build_sensor_features,
            _ENRICH_FEATURE_NAMES,
            _SENSOR_FEATURE_NAMES,
        )
        print("  [HybridCNN] Construction des features enrichies + capteurs...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            enrich_df = build_extended_features(steps_df)
            sensor_df = build_sensor_features(steps_df, ds_raw)

        # Fusionner features enrichies + capteurs
        enrich_df = enrich_df.set_index("step_id")
        sensor_df = sensor_df.set_index("step_id")
        all_tab_df = enrich_df.join(sensor_df, rsuffix="_sensor")

        tab_cols = [c for c in _ENRICH_FEATURE_NAMES + _SENSOR_FEATURE_NAMES
                    if c in all_tab_df.columns]
        X_tab = all_tab_df.reindex(columns=tab_cols).values.astype(np.float32)
        print(f"  [HybridCNN] {len(tab_cols)} features tabulaires pour le CNN hybride")

        cv_res, hybrid_preds_df = run_hybrid_cnn_cv(X_raw, X_tab, y, groups, steps_df)
        all_cv_results.extend(cv_res)

    # ── 4. Export CSV résultats ──────────────────────────────────────────
    print("\n[4/6] Export des résultats CSV...")
    df_cv = pd.DataFrame(all_cv_results)
    cv_path = _STEP_OUT_DIR / "model_step_comparison_cv.csv"
    df_cv.to_csv(cv_path, index=False)
    print(f"  -> {cv_path}")

    summary = (
        df_cv.groupby(["model", "level"])[["Accuracy", "Balanced_Acc", "F1-Score", "ROC-AUC"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("Balanced_Acc_mean", ascending=False)
    summ_path = _STEP_OUT_DIR / "phase_step_baseline_summary.csv"
    summary.to_csv(summ_path)
    print(f"  -> {summ_path}")

    print("\nRésumé (niveau sujet, Balanced_Acc) :")
    subj_summ = (
        df_cv[df_cv["level"] == "subject"]
        .groupby("model")[["Accuracy", "Balanced_Acc", "ROC-AUC"]]
        .mean()
        .sort_values("Balanced_Acc", ascending=False)
    )
    print(subj_summ.round(4).to_string())

    # ── 5. Table focus ───────────────────────────────────────────────────
    print("\n[5/6] Construction de la table focus (pas malades + asymétriques)...")
    if rf_preds_df is not None and not rf_preds_df.empty:
        focus = build_focus_table(steps_df, features_df, rf_preds_df)
        focus_path = _STEP_OUT_DIR / "focus_malades_premiers_pas.csv"
        focus.to_csv(focus_path, index=False)

        n_mismatch = (focus["anomaly_type"] == "pd_mismatch").sum()
        n_asym     = (focus["anomaly_type"] == "high_asym").sum()
        n_both     = (focus["anomaly_type"] == "both").sum()
        print(f"  Pas focalisés : {len(focus)} total")
        print(f"    pd_mismatch : {n_mismatch}  (CO prédit PD par RF)")
        print(f"    high_asym   : {n_asym}  (asymétrie > seuil)")
        print(f"    both        : {n_both}  (cumul)")
        print(f"  -> {focus_path}")
    else:
        print("  [WARN] RF predictions manquantes — table focus ignorée.")
        focus = pd.DataFrame()

    # ── 6. Graphiques ────────────────────────────────────────────────────
    print("\n[6/6] Génération des graphiques...")
    generate_comparison_plots(df_cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _plot_asymmetry_distribution(focus)
    print(f"  -> {_STEP_FIG_DIR}")

    print("\n" + "=" * 60)
    print("  TERMINÉ — output/etude_du_pas/")
    print("=" * 60)

    return hybrid_preds_df if "hybrid_preds_df" in dir() else None


if __name__ == "__main__":
    run_step_comparison()
