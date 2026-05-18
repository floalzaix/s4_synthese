"""
[ROLE]
Ce fichier génère des visualisations intermédiaires et avancées du cycle de marche.

[RESPONSIBILITY]
- Construire le profil moyen du pas (0-100% du cycle) pour PD et CO.
- Générer des heatmaps statiques de l'asymétrie sur le cycle de pas.
- Comparer les distributions des features clés par étude et par groupe.

[INPUTS]
- Fichiers de signaux bruts.
- Matrice de caractéristiques de la Phase 4.

[OUTPUTS]
- output/figures/gait_profiles/*.png
- output/figures/asymmetry_heatmaps/*.png

[ASSUMPTIONS]
- Un pas est interpolé sur 100 points pour normalisation temporelle.
- Session 01 uniquement.

[RISKS]
- L'interpolation peut lisser des pics très brefs.

[DEPENDENCIES]
- matplotlib
- seaborn
- scipy
- project.config
- project.features
- project.load
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from project.config import OUTPUT_DIR, SESSION
from project.features import build_feature_matrix, segment_steps
from project.load import load_dataset_index, load_signal_file

_PROFILE_DIR = OUTPUT_DIR / "figures" / "gait_profiles"
_HEATMAP_DIR = OUTPUT_DIR / "figures" / "asymmetry_heatmaps"


def _normalize_step(
    signal: np.ndarray, start: int, end: int, n_points: int = 100
) -> np.ndarray:
    """
    @brief Interpole un segment de signal sur un nombre fixe de points (0-100%).
    """
    segment = signal[start:end]
    if len(segment) < 2:
        return np.full(n_points, np.nan)
    x = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, n_points)
    f = interp1d(
        x, segment, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return f(x_new)


def generate_mean_step_profiles_and_heatmaps():
    idx = load_dataset_index()
    subset = idx[(idx["session"] == SESSION) & (idx["has_signal"] == True)].copy()

    pd_profiles = []
    co_profiles = []

    asym_matrix_pd = []
    asym_matrix_co = []

    print("Traitement des signaux pour le recuit temporel (0-100%)...")

    for _, row in subset.iterrows():
        try:
            sig = load_signal_file(row["filepath"])
        except Exception:
            continue

        L = sig["total_L"].values
        R = sig["total_R"].values

        # On va baser le cycle sur les pas gauches pour l'exemple
        seg_L = segment_steps(L)
        if not seg_L:
            continue

        subject_profiles = []
        subject_asym = []

        for s, e in seg_L:
            norm_L = _normalize_step(L, s, e)
            norm_R = _normalize_step(R, s, e)

            if np.isnan(norm_L).any() or np.isnan(norm_R).any():
                continue

            # Asymetrie dynamique sur ce pas: (R - L) / (R + L + eps)
            asym = (norm_R - norm_L) / (norm_R + norm_L + 1e-9)

            subject_profiles.append(norm_L)
            subject_asym.append(asym)

        if subject_profiles:
            # Profil moyen du sujet
            mean_prof = np.mean(subject_profiles, axis=0)
            mean_asym = np.mean(subject_asym, axis=0)

            if row["group"] == "PD":
                pd_profiles.append(mean_prof)
                asym_matrix_pd.append(mean_asym)
            else:
                co_profiles.append(mean_prof)
                asym_matrix_co.append(mean_asym)

    # --- 1. Mean Step Profile ---
    print("Génération du profil moyen...")
    plt.figure(figsize=(10, 6))

    x_axis = np.linspace(0, 100, 100)

    for group_name, profiles, color in [
        ("PD", pd_profiles, "salmon"),
        ("CO", co_profiles, "skyblue"),
    ]:
        mat = np.array(profiles)
        mean_curve = np.mean(mat, axis=0)
        std_curve = np.std(mat, axis=0)

        plt.plot(
            x_axis, mean_curve, color=color, label=f"{group_name} (mean)", linewidth=2
        )
        plt.fill_between(
            x_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.3,
        )

    plt.title("Profil moyen de la phase d'appui (Force totale - Normalisée 0-100%)")
    plt.xlabel("% de la phase d'appui")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(alpha=0.3)

    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(_PROFILE_DIR / "mean_stance_profile.png", dpi=150)
    plt.close()

    # --- 2. Heatmap Asymmetry ---
    print("Génération des heatmaps d'asymétrie...")
    full_matrix = np.vstack([asym_matrix_pd, asym_matrix_co])

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        full_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        cbar_kws={"label": "Asymétrie (R-L)/(R+L)"},
    )

    plt.axhline(len(asym_matrix_pd), color="black", linewidth=2)
    ax.set_yticks([])

    plt.title(
        f"Heatmap de l'Asymétrie dynamique sur la phase d'appui\n(Haut: PD n={len(asym_matrix_pd)} | Bas: CO n={len(asym_matrix_co)})"
    )
    plt.xlabel("% de la phase d'appui")
    plt.ylabel("Sujets")

    _HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(_HEATMAP_DIR / "dynamic_asymmetry_heatmap.png", dpi=150)
    plt.close()


def generate_stratified_distributions():
    print("Génération des distributions stratifiées par étude...")
    df = build_feature_matrix(session=SESSION)

    features_to_plot = ["std_asym", "asym_swing", "cv_swing_L", "n_steps"]

    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    for feat in features_to_plot:
        if feat not in df.columns:
            continue

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=df,
            x="study",
            y=feat,
            hue="group",
            split=True,
            inner="quart",
            palette={"PD": "salmon", "CO": "skyblue"},
        )
        plt.title(f"Distribution de {feat} stratifiée par Étude et Groupe")
        plt.xlabel("Étude")
        plt.ylabel(feat)
        plt.legend(title="Groupe")
        plt.tight_layout()
        plt.savefig(_PROFILE_DIR / f"stratified_{feat}.png", dpi=150)
        plt.close()


def main():
    generate_mean_step_profiles_and_heatmaps()
    generate_stratified_distributions()
    print(
        "Terminé ! Les figures sont dans output/figures/gait_profiles/ et asymmetry_heatmaps/"
    )


if __name__ == "__main__":
    main()
