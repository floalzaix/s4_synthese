"""
Chargement du dataset PhysioNet « Gait in Parkinson's Disease v1.0.0 ».

Anomalies documentées (non corrigées dans les données brutes) :
- Juc010 : entrée présente dans demographics.xls mais aucun fichier signal
  correspondant n'existe sur le disque. filepath=None dans l'index.
- Height (meters) : la colonne contient des valeurs en cm pour l'étude Ju
  (160–185) et en mètres pour Ga/Si (1.50–1.95). La colonne `height_m` du
  DataFrame retourné par load_demographics() applique la correction explicite.
"""

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
    Lit demographics.xls et retourne un DataFrame propre, indexé sur ID.

    Colonnes ajoutées :
      height_m  — hauteur en mètres pour tous les sujets (corrige l'incohérence
                  d'unité pour l'étude Ju dont les valeurs originales sont en cm).

    Toutes les colonnes originales de demographics.xls sont conservées.
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
    Parcourt le dossier dataset et retourne la liste des fichiers signal.

    Chaque entrée est un dict :
      subject_id  — ex. "GaPt03"
      session     — ex. "01"
      walk_type   — "normal" | "normal_2" | "ras_N" | "dual_task" | "unknown_N"
      filepath    — chemin absolu (Path)

    Les fichiers non-signal (demographics.txt, format.txt, SHA256SUMS.txt)
    sont ignorés.
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
    Lit un fichier signal .txt (TSV sans en-tête, 19 colonnes, 100 Hz).

    Retourne un DataFrame avec les colonnes :
      time, L1–L8, R1–R8, total_L, total_R
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
    Retourne une table complète (sujet × session) avec métadonnées.

    Colonnes garanties :
      subject_id, session, walk_type, filepath  — depuis les fichiers signal
      study, group, + toutes les colonnes de demographics  — depuis demographics.xls
      height_m  — hauteur normalisée en mètres

    Une ligne par (sujet, session). Le sujet Juc010 (dans demographics mais sans
    fichier signal) apparaît avec filepath=None.

    Sujets présents dans les fichiers mais absents de demographics sont exclus
    (aucun cas connu dans cette version du dataset).
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


# ---------------------------------------------------------------------------
# Exemple d'utilisation (exécutable directement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== load_demographics() ===")
    demo = load_demographics()
    print(f"  {len(demo)} sujets, {len(demo.columns)} colonnes")
    print(f"  Colonnes : {list(demo.columns)}")
    print(
        f"  PD : {(demo['Group'] == 'PD').sum()}, CO : {(demo['Group'] == 'CO').sum()}"
    )
    print(f"  HoehnYahr manquant : {demo['HoehnYahr'].isna().sum()}")
    print(f"  UPDRS manquant     : {demo['UPDRS'].isna().sum()}")
    print(f"  UPDRSM manquant    : {demo['UPDRSM'].isna().sum()}")
    print()

    print("=== list_signal_files() ===")
    files = list_signal_files()
    print(f"  {len(files)} fichiers signal")
    by_study = {}
    for f in files:
        s = f["subject_id"][:2]
        by_study[s] = by_study.get(s, 0) + 1
    print(f"  Par étude : {by_study}")
    print()

    print("=== load_signal_file() — exemple GaCo01_01.txt ===")
    import pathlib

    root = _resolve_root()
    sig = load_signal_file(root / "GaCo01_01.txt")
    print(f"  Shape : {sig.shape}")
    print(f"  Durée : {sig['time'].max():.1f} s  ({len(sig)} échantillons @ 100 Hz)")
    print(f"  Colonnes : {list(sig.columns)}")
    print()

    print("=== load_dataset_index() ===")
    idx = load_dataset_index()
    print(f"  {len(idx)} lignes (sujets × sessions)")
    print(
        f"  Sans fichier signal (filepath=NaN) : "
        f"{idx['filepath'].isna().sum()} sujet(s)"
    )
    print(
        f"  Aperçu :\n{idx[['subject_id', 'session', 'walk_type', 'Group', 'HoehnYahr']].head(8).to_string(index=False)}"
    )
