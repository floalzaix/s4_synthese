##
# @file load.py
# @brief Fonctions de chargement et de normalisation du dataset PhysioNet « Gait in Parkinson's Disease v1.0.0 ».
# @details Gère les données démographiques (.xls) et les signaux temporels (.txt).
#

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

_DEFAULT_DATASET_NAME = "gait-in-parkinsons-disease-1.0.0"


def _resolve_root(root_dir: Optional[str | Path] = None) -> Path:
    """
    @brief Résout le chemin racine du dataset.
    @param root_dir Chemin optionnel vers la racine du dataset.
    @return Path Chemin résolu.
    """
    if root_dir is not None:
        return Path(root_dir)
    # Remonte de project/ vers la racine du dépôt, puis datasets/
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    return repo_root / "datasets" / _DEFAULT_DATASET_NAME


# ---------------------------------------------------------------------------
# load_demographics
# ---------------------------------------------------------------------------


def load_demographics(root_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """
    @brief Lit demographics.xls et retourne un DataFrame propre, indexé sur ID.
    @details Normalise la hauteur en mètres pour tous les sujets.
    @param root_dir Chemin optionnel vers la racine du dataset.
    @return pd.DataFrame Données démographiques nettoyées.
    """
    root = _resolve_root(root_dir)
    xls_path = root / "demographics.xls"
    if not xls_path.exists():
        raise FileNotFoundError(f"demographics.xls introuvable : {xls_path}")

    df = pd.read_excel(xls_path, engine="xlrd")

    # Supprimer les lignes entièrement vides (bas du fichier)
    df = df.dropna(how="all").copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df = df[df["ID"] != "nan"].reset_index(drop=True)

    # Normalisation hauteur : Ju en cm → m, Ga/Si déjà en m
    height_col = "Height (meters)"
    df["height_m"] = df.apply(
        lambda row: (
            row[height_col] / 100.0 if row["Study"] == "Ju" else row[height_col]
        ),
        axis=1,
    )

    df = df.set_index("ID")
    return df


# ---------------------------------------------------------------------------
# list_signal_files
# ---------------------------------------------------------------------------

_SESSION_PATTERN = re.compile(r"^([A-Za-z]{2}[A-Za-z]{2}\d{2})_(\d+)\.txt$")

_WALK_TYPE_MAP = {
    "01": "normal",
    "02": "normal_2",
    "10": "dual_task",
}


def _session_to_walk_type(session: str) -> str:
    """
    @brief Mappe une session vers un type de marche.
    @param session Identifiant de la session.
    @return str Nom du type de marche.
    """
    if session in _WALK_TYPE_MAP:
        return _WALK_TYPE_MAP[session]
    try:
        n = int(session)
        if 3 <= n <= 7:
            return f"ras_{n}"
    except ValueError:
        pass
    return f"unknown_{session}"


def list_signal_files(root_dir: Optional[str | Path] = None) -> list[dict]:
    """
    @brief Parcourt le dossier dataset et retourne la liste des fichiers signal.
    @details Filtre les fichiers non-signal et extrait l'ID sujet et le type de session.
    @param root_dir Chemin optionnel vers la racine du dataset.
    @return list[dict] Liste des métadonnées des fichiers de signaux.
    """
    root = _resolve_root(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dossier dataset introuvable : {root}")

    records = []
    for fname in sorted(root.iterdir()):
        if fname.suffix != ".txt":
            continue
        m = _SESSION_PATTERN.match(fname.name)
        if m is None:
            continue
        subject_id, session = m.group(1), m.group(2)
        records.append(
            {
                "subject_id": subject_id,
                "session": session,
                "walk_type": _session_to_walk_type(session),
                "filepath": fname.resolve(),
            }
        )
    return records


# ---------------------------------------------------------------------------
# load_signal_file
# ---------------------------------------------------------------------------

_SIGNAL_COLUMNS = [
    "time",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "total_L",
    "total_R",
]


def load_signal_file(filepath: str | Path) -> pd.DataFrame:
    """
    @brief Lit un fichier signal .txt (TSV sans en-tête, 19 colonnes, 100 Hz).
    @param filepath Chemin du fichier à charger.
    @return pd.DataFrame Signaux temporels.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=_SIGNAL_COLUMNS,
        dtype=float,
    )
    return df


# ---------------------------------------------------------------------------
# load_dataset_index
# ---------------------------------------------------------------------------


def load_dataset_index(root_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """
    @brief Retourne une table complète (sujet × session) avec métadonnées.
    @details Fusionne les données démographiques et les chemins de fichiers.
    @param root_dir Chemin optionnel vers la racine du dataset.
    @return pd.DataFrame Index complet du dataset.
    """
    demo = load_demographics(root_dir)
    signals = list_signal_files(root_dir)

    signals_df = pd.DataFrame(signals)

    # Joint sur subject_id == ID (index de demo)
    index = demo.reset_index().merge(
        signals_df,
        left_on="ID",
        right_on="subject_id",
        how="left",
    )

    # Renommer pour cohérence
    index = index.rename(columns={"ID": "subject_id_demo"})
    index["subject_id"] = index["subject_id"].fillna(index["subject_id_demo"])

    index["has_signal"] = index["filepath"].notna()
    index["study"] = index["Study"]
    index["group"] = index["Group"]

    # Réordonner les colonnes prioritaires en tête
    priority = [
        "subject_id",
        "session",
        "walk_type",
        "filepath",
        "has_signal",
        "study",
        "group",
        "Study",
        "Group",
        "Subjnum",
        "Gender",
        "Age",
        "Height (meters)",
        "height_m",
        "Weight (kg)",
        "HoehnYahr",
        "UPDRS",
        "UPDRSM",
        "TUAG",
        "Speed_01 (m/sec)",
        "Speed_10",
    ]
    remaining = [c for c in index.columns if c not in priority + ["subject_id_demo"]]
    index = index[priority + remaining]

    return index.reset_index(drop=True)
