"""
[ROLE]
Visualisations avancées : Profils de marche et distributions stratifiées.

[RESPONSIBILITY]
- Construire le profil moyen du pas consolidé PD vs CO.
- Générer des heatmaps d'asymétrie dynamique.
- Visualiser la variabilité inter-études avec des facettes.

[OUTPUTS]
- output/figures/gait_profiles/*.png
- output/figures/asymmetry_heatmaps/*.png
"""

##
# @file viz_advanced.py
# @brief Visualisations avancées de la marche.
# @details Génère des profils de force moyens, des heatmaps d'asymétrie et des analyses stratifiées.
#

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from project.config import ASYM_FIG_DIR, GAIT_FIG_DIR, SESSION
from project.features import build_feature_matrix, segment_steps
from project.load import load_dataset_index, load_signal_file
from project.viz_utils import (
    FIG_STD,
    FIG_WIDE,
    PALETTE,
    clean_label,
    save_fig,
    setup_style,
)

setup_style()


def _normalize(sig, s, e, n=100):
    """
    @brief Normalise un segment de signal temporel à une longueur fixe.
    @param sig Signal brut.
    @param s Index de début.
    @param e Index de fin.
    @param n Nombre de points cibles (défaut 100).
    @return np.ndarray Signal normalisé.
    """
    seg = sig[s:e]
    if len(seg) < 2:
        return np.full(n, np.nan)
    f = interp1d(
        np.linspace(0, 1, len(seg)),
        seg,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(np.linspace(0, 1, n))


def generate_atlas_profiles():
    """
    @brief Génère les profils de force consolidés et les heatmaps d'asymétrie.
    """
    print(f"--- Optimisation Profils de Marche - Session: {SESSION} ---")
    idx = load_dataset_index()
    subset = idx[(idx["session"] == SESSION) & (idx["has_signal"])]

    pd_profs, co_profs = [], []
    pd_asym, co_asym = [], []

    for _, row in subset.iterrows():
        try:
            sig = load_signal_file(row["filepath"])
        except Exception:
            continue
        L, R = sig["total_L"].values, sig["total_R"].values
        segs = segment_steps(L)
        if not segs:
            continue

        subj_p, subj_a = [], []
        for s, e in segs:
            nL, nR = _normalize(L, s, e), _normalize(R, s, e)
            if np.isnan(nL).any() or np.isnan(nR).any():
                continue
            subj_p.append(nL)
            subj_a.append((nR - nL) / (nR + nL + 1e-9))

        if subj_p:
            if row["group"] == "PD":
                pd_profs.append(np.mean(subj_p, axis=0))
                pd_asym.append(np.mean(subj_a, axis=0))
            else:
                co_profs.append(np.mean(subj_p, axis=0))
                co_asym.append(np.mean(subj_a, axis=0))

    # 1. Consolidated Profile
    fig, ax = plt.subplots(figsize=FIG_STD)
    x = np.linspace(0, 100, 100)
    for grp, data, col in [
        ("PD", pd_profs, PALETTE["PD"]),
        ("CO", co_profs, PALETTE["CO"]),
    ]:
        if not data:
            continue
        mat = np.array(data)
        m, s = np.mean(mat, axis=0), np.std(mat, axis=0)
        ax.plot(x, m, color=col, label=f"{grp} (mean)", lw=2.5)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)
    ax.set_title("Signature Moyenne de Force (Phase d'appui)")
    ax.set_xlabel("% du Cycle")
    ax.set_ylabel("Force (N)")
    ax.legend()
    save_fig(fig, GAIT_FIG_DIR, "adv_gait_signature")

    # 2. Heatmap
    if pd_asym and co_asym:
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        sns.heatmap(
            np.vstack([pd_asym, co_asym]),
            cmap="coolwarm",
            center=0,
            vmin=-0.4,
            vmax=0.4,
            ax=ax,
            cbar_kws={"label": "Asymmetry (R-L)/(R+L)"},
        )
        ax.axhline(len(pd_asym), color="black", lw=3)
        ax.set_title("Heatmap d'Asymétrie Dynamique (PD vs CO)")
        ax.set_yticks([])
        save_fig(fig, ASYM_FIG_DIR, "adv_asymmetry_heatmap")


def generate_faceted_distributions(df=None):
    """
    @brief Génère des boxplots stratifiés par étude pour les features clés.
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    if df is None:
        df = build_feature_matrix(session=SESSION)
    feats = ["std_asym", "asym_swing", "cv_swing_L", "n_steps"]
    for f in feats:
        if f not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        sns.boxplot(data=df, x="study", y=f, hue="group", palette=PALETTE, ax=ax)
        ax.set_title(f"Variabilité Inter-Étude : {clean_label(f)}")
        ax.set_ylabel(clean_label(f))
        save_fig(fig, GAIT_FIG_DIR, f"adv_stratified_{f}")


def main(df=None):
    """
    @brief Point d'entrée principal pour les visualisations avancées.
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    generate_atlas_profiles()
    generate_faceted_distributions(df=df)


if __name__ == "__main__":
    main()
