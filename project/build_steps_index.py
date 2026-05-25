"""
build_steps_index.py — Construit steps.csv (et steps_features.csv) pour GaitPDB.

Logique de construction :
  - Chaque ligne de steps.csv correspond à un pas (stance phase) d'un seul pied.
  - Le signal brut n'est PAS stocké dans le CSV : seuls les indices (sample_start,
    sample_end) sont conservés. Un DataLoader PyTorch/TensorFlow recharge la tranche
    du fichier source à la volée, ce qui évite de dupliquer ~Go de signaux.
  - Un stride_id associe chaque pas gauche au pas droit le plus proche dans le temps
    (appariement glouton par midpoint), permettant des analyses de symétrie par cycle.
  - quality_flag permet de filtrer les pas tronqués ou trop courts sans les supprimer :
    le filtre reste à la charge du DataLoader / de l'entraînement.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from project.config import OUTPUT_DIR, REPO_ROOT
from project.features import _EPS, _FS, _trapz, segment_steps
from project.load import load_dataset_index, load_signal_file

# Un pas est considéré "de bord" si son début ou sa fin est à moins de N samples
# du début ou de la fin du signal (pas potentiellement tronqué).
_EDGE_MARGIN_SAMPLES: int = 5

# Garde défensif : segment_steps filtre déjà e-s < 10, mais on conserve ce seuil
# pour que quality_flag reste cohérent si segment_steps est modifié ultérieurement.
_MIN_STEP_SAMPLES: int = 10   # = 0.1 s @ 100 Hz

# Pas > 2.0 s : quasi-systématiquement un artefact (fusion de deux appuis ou pause
# d'enregistrement). Taggués "too_long" mais conservés dans le CSV ; le DataLoader
# doit les exclure pour éviter des fenêtres brutes corrompues.
_MAX_STEP_SAMPLES: int = 200  # = 2.0 s @ 100 Hz

# Distance maximale (en samples) entre le midpoint de deux pas pour être appariés
# en un même stride. Au-delà, le pas reste non apparié (stride_id = NaN).
# 150 samples = 1.5 s @ 100 Hz — couvre les arythmies sévères sans faux appariements.
_MAX_STRIDE_DISTANCE_SAMPLES: float = 150.0

# Colonnes finales de steps.csv (ordre canonique).
_STEPS_COLUMNS: list[str] = [
    "step_id",
    "subject_id",
    "group",
    "study",
    "session",
    "walk_type",
    "foot",
    "step_idx",
    "stride_id",
    "source_file",
    "sample_start",
    "sample_end",
    "duration_s",
    "peak_force",
    "auc_force",
    "quality_flag",
]


# ---------------------------------------------------------------------------
# Qualité du pas
# ---------------------------------------------------------------------------


def _quality_flag(start: int, end: int, n_samples: int) -> str:
    """
    Retourne un label de qualité pour un pas donné.

    Ordre de priorité : too_short > too_long > edge_start > edge_end > ok.
    Tous les flags sont conservés dans steps.csv ; le filtrage est laissé
    à l'étape aval (DataLoader, preprocessing).

    too_short : pas inférieur à _MIN_STEP_SAMPLES (0.1 s) — invalide.
    too_long  : pas supérieur à _MAX_STEP_SAMPLES (2.0 s) — artefact de fusion
                ou de pause d'enregistrement (validé par QC visuel).
    edge_*    : pas potentiellement tronqué en début/fin de signal.
    """
    if end - start < _MIN_STEP_SAMPLES:
        return "too_short"
    if end - start > _MAX_STEP_SAMPLES:
        return "too_long"
    if start <= _EDGE_MARGIN_SAMPLES:
        return "edge_start"
    if end >= n_samples - _EDGE_MARGIN_SAMPLES:
        return "edge_end"
    return "ok"


# ---------------------------------------------------------------------------
# Segmentation d'un pied
# ---------------------------------------------------------------------------


def _steps_for_foot(signal: np.ndarray, foot: str) -> list[dict]:
    """
    Segmente un signal de force 1D et retourne une liste de dicts par pas.

    Chaque dict contient les champs bruts du pas ainsi qu'un champ interne
    '_midpoint' (en samples) utilisé pour l'appariement L/R en stride.
    Ce champ n'est pas exporté dans steps.csv.
    """
    segments = segment_steps(signal)
    n_samples = len(signal)
    records = []

    for step_idx, (start, end) in enumerate(segments):
        step_sig = signal[start:end]
        # dx=1/_FS : intégration en secondes -> auc_force en [force·s], comparable
        # entre pas de durées différentes et entre datasets à fréquences différentes.
        records.append(
            {
                "foot": foot,
                "step_idx": step_idx,
                "sample_start": int(start),
                "sample_end": int(end),
                "duration_s": round((end - start) / _FS, 4),
                "peak_force": round(float(np.max(step_sig)), 4),
                "auc_force": round(float(_trapz(step_sig, dx=1.0 / _FS)), 4),
                "quality_flag": _quality_flag(start, end, n_samples),
                "_midpoint": (start + end) / 2.0,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Appariement L/R -> stride_id
# ---------------------------------------------------------------------------


def _assign_stride_ids(steps: list[dict]) -> list[dict]:
    """
    Assigne un stride_id à chaque pas par appariement glouton L↔R.

    Algorithme :
      1. Séparer les pas gauches et droits, triés par midpoint.
      2. Pour chaque pas L (dans l'ordre temporel), trouver le pas R non encore
         apparié dont le midpoint est le plus proche.
      3. Les deux pas reçoivent le même stride_id (= index du pas L dans la séquence).
      4. Les pas sans correspondant reçoivent stride_id = NaN.

    Complexité O(n²) sur la longueur de la séquence (max ~100 pas/enregistrement),
    ce qui est négligeable. La robustesse est bonne pour la marche normale et
    dégradée (arythmie Parkinson) car on cherche le plus proche, pas le suivant.
    """
    left = sorted([s for s in steps if s["foot"] == "L"], key=lambda s: s["_midpoint"])
    right = sorted([s for s in steps if s["foot"] == "R"], key=lambda s: s["_midpoint"])

    # Initialiser tous les stride_id à NaN
    for s in steps:
        s["stride_id"] = np.nan

    if not left or not right:
        return steps

    matched_right_indices: set[int] = set()

    for stride_id, l_step in enumerate(left):
        best_idx: int | None = None
        best_dist = float("inf")

        for r_idx, r_step in enumerate(right):
            if r_idx in matched_right_indices:
                continue
            dist = abs(l_step["_midpoint"] - r_step["_midpoint"])
            if dist < best_dist:
                best_dist = dist
                best_idx = r_idx

        # Ne pas apparier si les midpoints sont trop éloignés (ex. pas L isolé
        # en fin d'enregistrement après un arrêt prématuré du pied droit).
        if best_idx is not None and best_dist <= _MAX_STRIDE_DISTANCE_SAMPLES:
            l_step["stride_id"] = stride_id
            right[best_idx]["stride_id"] = stride_id
            matched_right_indices.add(best_idx)

    return steps


# ---------------------------------------------------------------------------
# Construction des enregistrements pour un fichier source
# ---------------------------------------------------------------------------


def _records_for_recording(
    row: pd.Series,
    sig: pd.DataFrame,
) -> list[dict]:
    """
    Construit toutes les lignes steps.csv pour un enregistrement (sujet × session).

    Le chemin source_file est stocké en relatif par rapport à REPO_ROOT pour
    garantir la portabilité du CSV entre machines (ex. Docker, autre OS).
    """
    subject_id: str = row["subject_id"]
    session: str = str(row["session"])
    walk_type: str = row["walk_type"]
    group: str = row["group"]
    study: str = row["study"]
    filepath = Path(row["filepath"])

    # Chemin POSIX (slashs) pour la portabilité cross-platform du CSV.
    # str() sur Windows produit des backslashs -> invalides sur Linux/macOS.
    try:
        source_file = filepath.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_file = filepath.as_posix()

    steps_L = _steps_for_foot(sig["total_L"].values, foot="L")
    steps_R = _steps_for_foot(sig["total_R"].values, foot="R")

    all_steps = _assign_stride_ids(steps_L + steps_R)

    records: list[dict] = []
    for step in all_steps:
        foot = step["foot"]
        step_idx = step["step_idx"]

        # Identifiant unique et lisible : "GaCo01_01_L_003"
        step_id = f"{subject_id}_{session}_{foot}_{step_idx:03d}"

        records.append(
            {
                "step_id": step_id,
                "subject_id": subject_id,
                "group": group,
                "study": study,
                "session": session,
                "walk_type": walk_type,
                "foot": foot,
                "step_idx": step_idx,
                "stride_id": step["stride_id"],  # float (NaN si non apparié)
                "source_file": source_file,
                "sample_start": step["sample_start"],
                "sample_end": step["sample_end"],
                "duration_s": step["duration_s"],
                "peak_force": step["peak_force"],
                "auc_force": step["auc_force"],
                "quality_flag": step["quality_flag"],
            }
        )

    return records


# ---------------------------------------------------------------------------
# Constructeur principal de steps.csv
# ---------------------------------------------------------------------------


def build_steps_index(
    root_dir=None,
    sessions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Parcourt tout le dataset et construit le DataFrame maître steps.csv.

    Paramètres
    ----------
    root_dir : chemin optionnel vers la racine du dataset (défaut : auto-détecté).
    sessions : liste de sessions à inclure (ex. ["01", "02"]).
               Si None, toutes les sessions avec fichier signal sont traitées.

    Retourne
    --------
    pd.DataFrame avec les colonnes de _STEPS_COLUMNS, une ligne par pas.
    """
    index = load_dataset_index(root_dir)
    mask = index["has_signal"]
    if sessions is not None:
        mask = mask & index["session"].isin(sessions)
    subset = index[mask].copy()

    all_records: list[dict] = []
    n_files = len(subset)

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        try:
            sig = load_signal_file(row["filepath"])
        except Exception as exc:
            warnings.warn(
                f"[{i}/{n_files}] Impossible de charger {row['filepath']} : {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        records = _records_for_recording(row, sig)
        all_records.extend(records)

        subj = row["subject_id"]
        sess = row["session"]
        n = len(records)
        print(f"  [{i:3d}/{n_files}] {subj} session {sess} -> {n} pas")

    if not all_records:
        warnings.warn("Aucun pas trouvé. Vérifier que le dataset est accessible.", RuntimeWarning)
        return pd.DataFrame(columns=_STEPS_COLUMNS)

    df = pd.DataFrame(all_records)[_STEPS_COLUMNS].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Fichier annexe : steps_features.csv (optionnel, modèles ML classiques)
# ---------------------------------------------------------------------------


def build_steps_features(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit steps_features.csv : features agrégées par pas pour les modèles ML classiques.

    Fichier secondaire, sans signaux bruts. Utilisé pour Random Forest / SVM qui
    n'ont pas besoin de la série temporelle complète.

    Colonnes produites (keyed sur step_id -> FK vers steps.csv) :
      - auc_norm      : AUC [force·s] / duration_s -> force moyenne (sans biais de durée)
      - peak_norm_rec : peak / max_total_signal_recording (invariant à la corpulence)
      - {foot}{1..8}_peak / _mean : stats par capteur individuel sur la fenêtre du pas
    """
    # Cache signal + max-par-enregistrement pour éviter O(n_steps × n_samples).
    # max_total est calculé une seule fois par fichier source, pas par pas.
    _signal_cache: dict[str, pd.DataFrame] = {}
    _max_cache: dict[str, dict[str, float]] = {}  # source_file -> {total_col: max}

    records: list[dict] = []

    for _, row in steps_df.iterrows():
        source_file: str = row["source_file"]

        if source_file not in _signal_cache:
            filepath = REPO_ROOT / Path(source_file)
            try:
                loaded = load_signal_file(filepath)
                _signal_cache[source_file] = loaded
                # Précalcul du max par colonne total_ pour tout l'enregistrement
                _max_cache[source_file] = {
                    col: float(loaded[col].max())
                    for col in ("total_L", "total_R")
                    if col in loaded.columns
                }
            except Exception as exc:
                warnings.warn(f"Impossible de charger {source_file} : {exc}", RuntimeWarning)
                continue

        sig = _signal_cache[source_file]
        foot: str = row["foot"]
        total_col = f"total_{foot}"
        start = int(row["sample_start"])
        end = int(row["sample_end"])

        step_sig = sig[total_col].values[start:end]
        duration_s: float = float(row["duration_s"])

        # AUC déjà en [force·s] (dx=1/_FS appliqué lors de la construction de steps.csv).
        # On divise par duration_s pour obtenir la force moyenne, comparable entre pas.
        auc_norm = float(_trapz(step_sig, dx=1.0 / _FS)) / (duration_s + _EPS)

        rec_max = _max_cache[source_file].get(total_col, 1.0)
        peak_norm_rec = float(np.max(step_sig)) / (rec_max + _EPS)

        record: dict = {
            "step_id": row["step_id"],
            "auc_norm": round(auc_norm, 4),
            "peak_norm_rec": round(peak_norm_rec, 4),
        }

        sensor_cols = [f"{foot}{i}" for i in range(1, 9)]
        for sc in sensor_cols:
            if sc in sig.columns:
                s_slice = sig[sc].values[start:end]
                record[f"{sc}_peak"] = round(float(np.max(s_slice)), 4)
                record[f"{sc}_mean"] = round(float(np.mean(s_slice)), 4)

        records.append(record)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main(
    root_dir=None,
    sessions: Optional[list[str]] = None,
    include_features: bool = True,
) -> None:
    """
    Construit et exporte steps.csv (et éventuellement steps_features.csv).

    Paramètres
    ----------
    root_dir        : chemin optionnel vers la racine du dataset.
    sessions        : sessions à traiter. None = toutes.
    include_features: si True, exporte aussi steps_features.csv.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BUILD STEPS INDEX")
    print("=" * 60)
    print(
        f"  Sessions : {'toutes' if sessions is None else ', '.join(sessions)}\n"
    )

    steps_df = build_steps_index(root_dir=root_dir, sessions=sessions)

    if steps_df.empty:
        print("Aucun pas produit. Arrêt.")
        return

    # Résumé qualité
    flag_counts = steps_df["quality_flag"].value_counts().to_dict()
    total = len(steps_df)
    n_ok = flag_counts.get("ok", 0)
    print(f"\nTotal pas : {total}")
    for flag, count in sorted(flag_counts.items()):
        pct = 100 * count / total
        print(f"  {flag:<15} : {count:5d}  ({pct:.1f}%)")

    steps_path = OUTPUT_DIR / "steps.csv"
    steps_df.to_csv(steps_path, index=False)
    print(f"\nExporté : {steps_path}")

    if include_features:
        print("\nConstruction des features pas-level...")
        feats_df = build_steps_features(steps_df)
        feats_path = OUTPUT_DIR / "steps_features.csv"
        feats_df.to_csv(feats_path, index=False)
        print(f"Exporté : {feats_path}  ({len(feats_df)} lignes)")

    print("\n" + "=" * 60)
    print("  TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
