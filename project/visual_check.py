"""
[ROLE]
Contrôle Qualité (QC) visuel de la segmentation par pas.

[RESPONSIBILITY]
- Générer des figures multi-vues (Long, Zoom, Overlay) pour inspecter la précision.
- Produire un résumé visuel de la qualité de segmentation.

[OUTPUTS]
- output/figures/segmentation/*.png
"""

##
# @file visual_check.py
# @brief Contrôle Qualité (QC) visuel de la segmentation par pas.
# @details Génère des atlas visuels pour vérifier la détection des phases d'appui et d'oscillation.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.config import OUTPUT_DIR, SEG_FIG_DIR, SESSION
from project.features import segment_steps
from project.load import load_dataset_index, load_signal_file
from project.viz_utils import FIG_WIDE, save_fig, setup_style

setup_style()


def plot_segmentation_atlas(sig: pd.DataFrame, sid: str):
    """
    @brief Génère une vue atlas pour un sujet.
    @param sig DataFrame des signaux temporels.
    @param sid Identifiant du sujet.
    """
    time = sig["time"].values
    L, R = sig["total_L"].values, sig["total_R"].values
    seg_L = segment_steps(L)
    seg_R = segment_steps(R)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Long view (30s)
    t30 = time <= 30
    axes[0].plot(time[t30], L[t30], color="tab:blue", label="L")
    axes[0].plot(time[t30], R[t30] + 1200, color="tab:green", label="R (offset)")
    for s, e in seg_L:
        if time[s] <= 30:
            axes[0].axvspan(time[s], time[e], color="tab:blue", alpha=0.1)
    for s, e in seg_R:
        if time[s] <= 30:
            axes[0].axvspan(time[s], time[e], color="tab:green", alpha=0.1)
    axes[0].set_title(f"Aperçu Segmentation - {sid}")
    axes[0].legend(loc="upper right")

    # 2. Zoom Precision (5s)
    tz = (time >= 10) & (time <= 15)
    axes[1].plot(time[tz], L[tz], marker=".", markersize=3, label="Force L")
    for s, e in seg_L:
        if 10 <= time[s] <= 15:
            axes[1].axvspan(time[s], time[e], color="tab:blue", alpha=0.2)
            axes[1].axvline(
                time[s],
                color="red",
                linestyle="--",
                alpha=0.6,
                label="Start" if s == seg_L[0][0] else "",
            )
            axes[1].axvline(
                time[e],
                color="darkred",
                linestyle="--",
                alpha=0.6,
                label="End" if e == seg_L[0][1] else "",
            )
    axes[1].set_title("Zoom Précision (Indices Début/Fin)")

    # 3. Overlay L vs R
    axes[2].plot(time[t30], L[t30], color="tab:blue", label="L")
    axes[2].plot(time[t30], R[t30], color="tab:green", alpha=0.6, label="R")
    axes[2].set_title("Superposition Force Gauche / Droite")
    axes[2].set_xlabel("Temps (s)")
    axes[2].legend()

    save_fig(fig, SEG_FIG_DIR, f"seg_check_{sid}")


def main():
    """
    @brief Exécute le processus de QC visuel sur une sélection de sujets.
    """
    print(f"--- Optimisation QC Segmentation - Session: {SESSION} ---")
    idx = load_dataset_index()
    targets = ["GaPt03", "JuPt01", "SiPt02", "GaCo01", "GaPt23"]
    subset = idx[idx["subject_id"].isin(targets) & (idx["session"] == SESSION)]

    records = []
    for _, row in subset.iterrows():
        sid = row["subject_id"]
        sig = load_signal_file(row["filepath"])
        plot_segmentation_atlas(sig, sid)

        sL, sR = (
            len(segment_steps(sig["total_L"].values)),
            len(segment_steps(sig["total_R"].values)),
        )
        records.append(
            {"sid": sid, "n_L": sL, "n_R": sR, "status": "OK" if sL > 10 else "WARN"}
        )

    pd.DataFrame(records).to_csv(OUTPUT_DIR / "segmentation_quality.csv", index=False)
    print(f"QC Terminé. Atlas dans {SEG_FIG_DIR}")


if __name__ == "__main__":
    main()
