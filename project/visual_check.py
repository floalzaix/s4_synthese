"""
[ROLE]
Ce fichier contient la validation visuelle et le contrôle qualité de la segmentation par pas.

[RESPONSIBILITY]
- Générer des visualisations des signaux de force avec les segments de pas détectés.
- Calculer des métriques de diagnostic sur la régularité des pas.
- Produire un rapport de qualité pour un échantillon de sujets.

[INPUTS]
- Fichiers de signaux bruts.
- Logique de segmentation de project.features.

[OUTPUTS]
- output/segmentation_viz/*.png : Figures de visualisation.
- output/segmentation_visual_check.csv : Diagnostic de qualité.

[ASSUMPTIONS]
- Fréquence d'échantillonnage de 100 Hz.
- Les colonnes total_L et total_R sont fiables.

[RISKS]
- La visualisation peut être encombrée si trop de pas sont affichés (zoom requis).

[DEPENDENCIES]
- matplotlib
- numpy
- pandas
- project.load
- project.features
- project.config
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.config import OUTPUT_DIR, SESSION
from project.features import segment_steps
from project.load import load_dataset_index, load_signal_file

_VIZ_DIR = OUTPUT_DIR / "segmentation_viz"
_FS = 100


def diagnostic_metrics(segments: list[tuple[int, int]]) -> dict:
    """
    @brief Calcule des métriques de diagnostic sur la segmentation.
    @param segments Liste des segments (start, end).
    @return dict Métriques de diagnostic.
    """
    if len(segments) < 2:
        return {
            "n_steps": len(segments),
            "mean_interval": np.nan,
            "std_interval": np.nan,
            "short_interval_ratio": np.nan,
            "long_interval_ratio": np.nan,
            "quality": "bad" if len(segments) == 0 else "warning",
        }

    # Intervalles entre les milieux de pas successifs
    mids = np.array([(s + e) / 2.0 for s, e in segments])
    intervals = np.diff(mids) / _FS

    mean_int = np.mean(intervals)
    std_int = np.std(intervals)

    # Seuils arbitraires pour la marche humaine normale
    # Un intervalle < 0.4s ou > 2.0s est suspect pour un pas complet
    short_ratio = np.mean(intervals < 0.4)
    long_ratio = np.mean(intervals > 2.0)

    quality = "good"
    if std_int / (mean_int + 1e-9) > 0.2:
        quality = "warning"
    if short_ratio > 0.1 or long_ratio > 0.1:
        quality = "warning"
    if len(segments) < 10:
        quality = "bad"

    return {
        "n_steps": len(segments),
        "mean_interval": round(float(mean_int), 3),
        "std_interval": round(float(std_int), 3),
        "short_interval_ratio": round(float(short_ratio), 3),
        "long_interval_ratio": round(float(long_ratio), 3),
        "quality": quality,
    }


def plot_segmentation(sig: pd.DataFrame, subject_id: str, study: str, group: str):
    """
    @brief Génère une figure de visualisation de la segmentation.
    """
    L = sig["total_L"].values
    R = sig["total_R"].values
    time = sig["time"].values

    seg_L = segment_steps(L)
    seg_R = segment_steps(R)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # On affiche les 20 premières secondes pour la clarté
    mask = time <= 20
    t_plot = time[mask]
    L_plot = L[mask]
    R_plot = R[mask]

    # Plot Jambe Gauche
    axes[0].plot(t_plot, L_plot, color="blue", alpha=0.7, label="Total L")
    for s, e in seg_L:
        if time[s] > 20:
            continue
        axes[0].axvspan(time[s], time[e], color="blue", alpha=0.2)
        axes[0].axvline(time[s], color="blue", linestyle="--", alpha=0.3)
    axes[0].set_title(f"Segmentation L - {subject_id} ({group}, {study})")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend()

    # Plot Jambe Droite
    axes[1].plot(t_plot, R_plot, color="green", alpha=0.7, label="Total R")
    for s, e in seg_R:
        if time[s] > 20:
            continue
        axes[1].axvspan(time[s], time[e], color="green", alpha=0.2)
        axes[1].axvline(time[s], color="green", linestyle="--", alpha=0.3)
    axes[1].set_title(f"Segmentation R - {subject_id} ({group}, {study})")
    axes[1].set_ylabel("Force (N)")
    axes[1].set_xlabel("Temps (s)")
    axes[1].legend()

    plt.tight_layout()
    _VIZ_DIR.mkdir(exist_ok=True, parents=True)
    fig_path = _VIZ_DIR / f"{subject_id}_segmentation.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def main():
    print(f"Démarrage de la validation visuelle (Session {SESSION})...")
    idx = load_dataset_index()

    # Sélection des sujets tests
    test_ids = ["GaPt03", "JuPt01", "SiPt02", "GaCo01", "JuCo01", "SiCo01"]

    subset = idx[idx["subject_id"].isin(test_ids) & (idx["session"] == SESSION)].copy()

    results = []
    for _, row in subset.iterrows():
        sid = row["subject_id"]
        study = row["study"]
        group = row["group"]

        print(f"  Traitement de {sid}...")
        sig = load_signal_file(row["filepath"])

        # Visualisation
        plot_segmentation(sig, sid, study, group)

        # Diagnostic
        diag_L = diagnostic_metrics(segment_steps(sig["total_L"].values))
        diag_R = diagnostic_metrics(segment_steps(sig["total_R"].values))

        results.append(
            {
                "subject_id": sid,
                "study": study,
                "group": group,
                "n_steps_L": diag_L["n_steps"],
                "n_steps_R": diag_R["n_steps"],
                "mean_int_L": diag_L["mean_interval"],
                "std_int_L": diag_L["std_interval"],
                "quality_L": diag_L["quality"],
                "quality_R": diag_R["quality"],
                "short_ratio_L": diag_L["short_interval_ratio"],
                "long_ratio_L": diag_L["long_interval_ratio"],
            }
        )

    df_res = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "segmentation_visual_check.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nDiagnostic sauvegardé dans {csv_path}")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
