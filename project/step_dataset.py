"""
step_dataset.py — Dataset pas-level pour GaitPDB.

Architecture lazy-loading : steps.csv indexe les pas (source_file + sample offsets),
les signaux bruts sont chargés à la volée depuis les .txt d'origine.

Politique de filtrage par défaut
---------------------------------
Exclus  : too_short (< 0.1 s, invalide), too_long (> 2.0 s, artefact de fusion)
Inclus  : ok, edge_start, edge_end (potentiellement tronqués mais utilisables)
Contrôle: paramètre include_edge=False pour exclure aussi les pas de bord.

Compatibilité framework
-----------------------
La classe implémente __len__ + __getitem__ (interface PyTorch Dataset, duck-typing).
TensorFlow : utiliser StepDataset.as_generator() avec tf.data.Dataset.from_generator().
scikit-learn : StepDataset.to_arrays() retourne (X, y, groups) prêts pour GroupKFold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd

from project.config import OUTPUT_DIR, REPO_ROOT

# Colonnes des 16 capteurs individuels (ordre fixe, stable entre appels)
_SENSOR_COLS: list[str] = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)]

# Flags systématiquement exclus (pas invalides ou artefacts confirmés)
_ALWAYS_EXCLUDE: frozenset[str] = frozenset({"too_short", "too_long"})

_LABEL_MAP: dict[str, int] = {"PD": 1, "CO": 0}

_DEFAULT_STEPS_CSV = OUTPUT_DIR / "steps.csv"


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _load_sensor_array(filepath: Path) -> np.ndarray:
    """
    Lit un fichier signal et retourne uniquement les 16 colonnes capteurs,
    sous forme de tableau numpy float32 de shape (N, 16).
    """
    from project.load import load_signal_file

    sig = load_signal_file(filepath)
    return sig[_SENSOR_COLS].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------


class StepDataset:
    """
    Dataset pas-level sur GaitPDB.

    Paramètres
    ----------
    steps_csv      : chemin vers steps.csv (défaut : output/steps.csv).
    include_edge   : si False, exclut aussi les pas edge_start / edge_end.
    channels_first : si True, retourne des tableaux de shape (16, T) au lieu de (T, 16).
    cache_signals  : si True, met en cache les signaux bruts en mémoire.
                     Utile pour entraînements multi-époques. ~200 Mo RAM pour GaitPDB complet.
    """

    def __init__(
        self,
        steps_csv: Optional[Path | str] = None,
        *,
        include_edge: bool = True,
        channels_first: bool = False,
        cache_signals: bool = False,
    ) -> None:
        csv_path = Path(steps_csv) if steps_csv is not None else _DEFAULT_STEPS_CSV
        if not csv_path.exists():
            raise FileNotFoundError(
                f"steps.csv introuvable : {csv_path}\n"
                "Exécuter d'abord : python -m project.build_steps_index"
            )

        df = pd.read_csv(csv_path)

        exclude_flags = set(_ALWAYS_EXCLUDE)
        if not include_edge:
            exclude_flags.update({"edge_start", "edge_end"})

        self.df: pd.DataFrame = df[~df["quality_flag"].isin(exclude_flags)].reset_index(drop=True)
        self.channels_first = channels_first
        self._cache_signals = cache_signals
        self._signal_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Interface PyTorch Dataset (duck-typing — pas besoin d'importer torch)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, dict]:
        """
        Retourne (signal, label, metadata) pour le pas à l'indice idx.

        signal   : np.ndarray float32, shape (T, 16) ou (16, T) si channels_first.
        label    : int  — 1 = PD, 0 = CO.
        metadata : dict avec subject_id, step_id, foot, stride_id, quality_flag,
                   session, walk_type, duration_s.
        """
        row = self.df.iloc[idx]

        signal = self._get_signal(
            source_file=str(row["source_file"]),
            start=int(row["sample_start"]),
            end=int(row["sample_end"]),
        )

        label = _LABEL_MAP.get(str(row["group"]), -1)

        metadata = {
            "subject_id":   str(row["subject_id"]),
            "group":        str(row["group"]),
            "step_id":      str(row["step_id"]),
            "foot":         str(row["foot"]),
            "stride_id":    row["stride_id"],  # float (NaN si non apparié)
            "quality_flag": str(row["quality_flag"]),
            "session":      str(row["session"]),
            "walk_type":    str(row["walk_type"]),
            "duration_s":   float(row["duration_s"]),
        }

        return signal, label, metadata

    # ------------------------------------------------------------------
    # Chargement signal
    # ------------------------------------------------------------------

    def _get_signal(self, source_file: str, start: int, end: int) -> np.ndarray:
        if self._cache_signals:
            if source_file not in self._signal_cache:
                self._signal_cache[source_file] = _load_sensor_array(
                    REPO_ROOT / Path(source_file)
                )
            full = self._signal_cache[source_file]
        else:
            full = _load_sensor_array(REPO_ROOT / Path(source_file))

        window = full[start:end]  # shape (T, 16)

        if self.channels_first:
            window = window.T  # shape (16, T)

        return window

    # ------------------------------------------------------------------
    # Compatibilité TensorFlow
    # ------------------------------------------------------------------

    def as_generator(self) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """
        Générateur Python pour tf.data.Dataset.from_generator().

        Exemple
        -------
        ds = StepDataset(cache_signals=True)
        output_signature = (
            tf.RaggedTensorSpec(shape=(None, 16), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        tf_ds = tf.data.Dataset.from_generator(
            ds.as_generator,
            output_signature=output_signature,
        )
        """
        for i in range(len(self)):
            signal, label, _ = self[i]
            yield signal, label

    # ------------------------------------------------------------------
    # Compatibilité scikit-learn
    # ------------------------------------------------------------------

    def to_arrays(
        self,
        pad_to: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (X, y, groups) pour GroupKFold ou cross_val_score.

        X      : float32, shape (n_steps, pad_to, 16) si pad_to fourni,
                 sinon liste de tableaux variable-length.
        y      : int32, shape (n_steps,).
        groups : object, shape (n_steps,) — subject_id pour GroupKFold.

        Note : avec pad_to=None, X est une liste numpy object array.
        Utiliser pad_to=max_len ou extraire des features avant d'appeler un modèle SK.
        """
        signals, labels, groups = [], [], []
        for i in range(len(self)):
            sig, lbl, meta = self[i]
            signals.append(sig)
            labels.append(lbl)
            groups.append(meta["subject_id"])

        y = np.array(labels, dtype=np.int32)
        g = np.array(groups, dtype=object)

        if pad_to is not None:
            X = np.zeros((len(signals), pad_to, 16), dtype=np.float32)
            for i, s in enumerate(signals):
                t = min(len(s), pad_to)
                X[i, :t] = s[:t]
        else:
            X = np.empty(len(signals), dtype=object)
            for i, s in enumerate(signals):
                X[i] = s

        return X, y, g

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n = len(self.df)
        flag_counts = self.df["quality_flag"].value_counts().to_dict()
        group_counts = self.df["group"].value_counts().to_dict()
        n_subjects = self.df["subject_id"].nunique()
        lines = [
            f"StepDataset — {n} pas, {n_subjects} sujets",
            f"  Groupes  : { {k: v for k, v in group_counts.items()} }",
            f"  Flags    : { {k: v for k, v in flag_counts.items()} }",
            f"  Shape    : {'(16, T)' if self.channels_first else '(T, 16)'}",
            f"  Cache    : {'activé' if self._cache_signals else 'désactivé'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SubjectSplitter — GroupKFold sans fuite de données
# ---------------------------------------------------------------------------


class SubjectSplitter:
    """
    Encapsule la logique de découpage sujet-level pour éviter la fuite de données.

    Garantit que tous les pas d'un même sujet sont dans le même fold.

    Exemple
    -------
    ds = StepDataset()
    splitter = SubjectSplitter(ds)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(n_splits=5)):
        train_ds = Subset(ds, train_idx)  # torch.utils.data.Subset
        val_ds   = Subset(ds, val_idx)
    """

    def __init__(self, dataset: StepDataset) -> None:
        self.groups: np.ndarray = dataset.df["subject_id"].to_numpy(dtype=object)

    def split(
        self, n_splits: int = 5
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_indices, val_indices) pour chaque fold.
        Nécessite scikit-learn.
        """
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=n_splits)
        dummy = np.zeros(len(self.groups))
        for train_idx, val_idx in gkf.split(dummy, groups=self.groups):
            yield train_idx, val_idx

    def train_val_split(
        self, val_subjects: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Découpe manuelle par liste de sujets (pour hold-out ou split prédéfini).
        """
        val_mask = np.isin(self.groups, val_subjects)
        return np.where(~val_mask)[0], np.where(val_mask)[0]


# ---------------------------------------------------------------------------
# Point d'entrée — résumé rapide
# ---------------------------------------------------------------------------


def main() -> None:
    print("Chargement de StepDataset (filtrage défaut : sans too_short / too_long)...")
    ds = StepDataset()
    print(ds.summary())

    print("\nExemple __getitem__(0) :")
    sig, lbl, meta = ds[0]
    print(f"  signal shape : {sig.shape}  dtype={sig.dtype}")
    print(f"  label        : {lbl}  (group={meta['group']})")
    print(f"  metadata     : {meta}")

    print("\nSubjectSplitter — 5-fold GroupKFold :")
    splitter = SubjectSplitter(ds)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(n_splits=5)):
        train_subj = set(splitter.groups[train_idx])
        val_subj   = set(splitter.groups[val_idx])
        assert train_subj.isdisjoint(val_subj), "Fuite détectée !"
        print(f"  Fold {fold+1} — train: {len(train_idx)} pas / {len(train_subj)} sujets  |  "
              f"val: {len(val_idx)} pas / {len(val_subj)} sujets")


if __name__ == "__main__":
    main()
