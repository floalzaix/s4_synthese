"""
Figure de synthèse : enveloppes PD vs CO normalisées par poids.

Superpose les profils de force de tous les pas (session 01, qualité ok)
rééchantillonnés sur 0-100 % de la phase d'appui et normalisés en N/kg.
Zone rouge = PD (Parkinson), zone verte = CO (contrôle sain).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaitpdb.config import OUTPUT_DIR, REPO_ROOT
from gaitpdb.load import load_demographics, load_signal_file
from gaitpdb.step_improvements import _butter_filter, _FS
from gaitpdb.viz.utils import save_fig, setup_style

_N_RESAMPLE = 101
_FIG_DIR = OUTPUT_DIR / "etude_du_pas" / "figures" / "improvements"


def _collect_normalised_steps() -> dict[str, np.ndarray]:
    """Charge tous les pas ok (session 01), filtre, normalise par poids, rééchantillonne."""
    steps = pd.read_csv(OUTPUT_DIR / "steps.csv")
    steps = steps[(steps["session"] == 1) & (steps["quality_flag"] == "ok")].copy()

    demo = load_demographics()
    weight_map = demo["Weight (kg)"].to_dict()

    sig_cache: dict[str, pd.DataFrame] = {}
    groups: dict[str, list[np.ndarray]] = {"PD": [], "CO": []}

    for _, row in steps.iterrows():
        group = row["group"]
        if group not in groups:
            continue
        subj = row["subject_id"]
        weight = weight_map.get(subj, np.nan)
        if np.isnan(weight) or weight <= 0:
            continue

        sf = str(row["source_file"])
        if sf not in sig_cache:
            try:
                sig_cache[sf] = load_signal_file(REPO_ROOT / Path(sf))
            except Exception:
                continue

        sig = sig_cache[sf]
        col = f"total_{row['foot']}"
        raw = sig[col].values[int(row["sample_start"]):int(row["sample_end"])].astype(np.float32)
        if len(raw) < 16:
            continue

        filtered = _butter_filter(raw)
        normalised = filtered / weight

        x_old = np.linspace(0, 100, len(normalised))
        x_new = np.linspace(0, 100, _N_RESAMPLE)
        resampled = np.interp(x_new, x_old, normalised)
        groups[group].append(resampled)

    return {g: np.array(curves) for g, curves in groups.items() if curves}


def plot_synthesis() -> None:
    setup_style()
    data = _collect_normalised_steps()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 100, _N_RESAMPLE)

    styles = {
        "PD": {"color": "#e74c3c", "label": "PD (Parkinson)"},
        "CO": {"color": "#27ae60", "label": "CO (Contrôle sain)"},
    }

    for group in ("CO", "PD"):
        if group not in data:
            continue
        mat = data[group]
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        s = styles[group]

        ax.fill_between(x, mean - std, mean + std, color=s["color"], alpha=0.25)
        ax.plot(x, mean, color=s["color"], linewidth=2.2, label=s["label"])

    ax.set_xlabel("Phase d'appui (%)")
    ax.set_ylabel("Force normalisée (N/kg)")
    ax.set_title(
        "Profils de force moyens — PD vs Contrôle\n"
        "(normalisé par poids, rééchantillonné 0-100 % stance)"
    )
    ax.legend(fontsize=11)

    n_pd = len(data.get("PD", []))
    n_co = len(data.get("CO", []))
    ax.annotate(
        f"PD : {n_pd} pas  |  CO : {n_co} pas",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color="grey",
    )

    save_fig(fig, _FIG_DIR, "step_zones_pd_co")
    print(f"  -> {_FIG_DIR / 'step_zones_pd_co.png'}")


if __name__ == "__main__":
    plot_synthesis()
