"""
validate_stance_threshold.py — Compare empiriquement la segmentation a threshold=0.05 vs 0.08.

Produit :
  - output/stance_threshold_comparison.csv     : stats par enregistrement
  - output/stance_threshold_suspects.csv       : cas ou les deux seuils divergent fortement
  - output/figures/segmentation_threshold/*.png : figures comparatives visuelles
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.config import OUTPUT_DIR, REPO_ROOT
from project.features import _FS, segment_steps
from project.load import load_dataset_index, load_signal_file
from project.viz_utils import save_fig, setup_style

setup_style()

# Seuils a comparer
_RATIO_LOW  = 0.05
_RATIO_HIGH = 0.08

# Sujets prioritaires issus du QC visuel
_FOCUS_SUBJECTS = ["GaPt23", "JuPt01", "GaCo16", "GaPt31", "GaPt15", "JuPt26"]

_OUT_FIG_DIR = OUTPUT_DIR / "figures" / "segmentation_threshold"

# Seuils de flag (memes que build_steps_index)
_MIN_STEP_S = 0.1   # too_short
_MAX_STEP_S = 2.0   # too_long
_EDGE_SAMPLES = 5


# ---------------------------------------------------------------------------
# Calcul des stats de segmentation pour un signal + un seuil
# ---------------------------------------------------------------------------

class SegStats(NamedTuple):
    n_steps:   int
    n_short:   int    # < 0.1 s
    n_long:    int    # > 2.0 s
    n_edge:    int    # edge_start ou edge_end
    dur_mean:  float
    dur_p50:   float
    dur_p95:   float
    dur_max:   float


def _seg_stats(signal: np.ndarray, ratio: float) -> SegStats:
    segs = segment_steps(signal, threshold_ratio=ratio)
    if not segs:
        return SegStats(0, 0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    n = len(signal)
    durs = np.array([(e - s) / _FS for s, e in segs])

    n_short = int((durs < _MIN_STEP_S).sum())
    n_long  = int((durs > _MAX_STEP_S).sum())
    n_edge  = sum(
        1 for s, e in segs
        if s <= _EDGE_SAMPLES or e >= n - _EDGE_SAMPLES
    )

    return SegStats(
        n_steps  = len(segs),
        n_short  = n_short,
        n_long   = n_long,
        n_edge   = n_edge,
        dur_mean = float(np.mean(durs)),
        dur_p50  = float(np.median(durs)),
        dur_p95  = float(np.percentile(durs, 95)),
        dur_max  = float(durs.max()),
    )


# ---------------------------------------------------------------------------
# Rapport comparatif par enregistrement
# ---------------------------------------------------------------------------

def _compare_recording(row: pd.Series, sig: pd.DataFrame) -> dict:
    L = sig["total_L"].values
    R = sig["total_R"].values

    lo_L = _seg_stats(L, _RATIO_LOW)
    hi_L = _seg_stats(L, _RATIO_HIGH)
    lo_R = _seg_stats(R, _RATIO_LOW)
    hi_R = _seg_stats(R, _RATIO_HIGH)

    total_lo = lo_L.n_steps + lo_R.n_steps
    total_hi = hi_L.n_steps + hi_R.n_steps

    return {
        "subject_id"     : row["subject_id"],
        "session"        : row["session"],
        "walk_type"      : row["walk_type"],
        "group"          : row["group"],
        "study"          : row["study"],
        # ---- seuil 0.05 ----
        "lo_n_steps"     : total_lo,
        "lo_n_long"      : lo_L.n_long  + lo_R.n_long,
        "lo_n_short"     : lo_L.n_short + lo_R.n_short,
        "lo_n_edge"      : lo_L.n_edge  + lo_R.n_edge,
        "lo_dur_mean"    : round((lo_L.dur_mean + lo_R.dur_mean) / 2, 4),
        "lo_dur_p50"     : round((lo_L.dur_p50  + lo_R.dur_p50)  / 2, 4),
        "lo_dur_p95"     : round((lo_L.dur_p95  + lo_R.dur_p95)  / 2, 4),
        "lo_dur_max"     : round(max(lo_L.dur_max, lo_R.dur_max), 4),
        # ---- seuil 0.08 ----
        "hi_n_steps"     : total_hi,
        "hi_n_long"      : hi_L.n_long  + hi_R.n_long,
        "hi_n_short"     : hi_L.n_short + hi_R.n_short,
        "hi_n_edge"      : hi_L.n_edge  + hi_R.n_edge,
        "hi_dur_mean"    : round((hi_L.dur_mean + hi_R.dur_mean) / 2, 4),
        "hi_dur_p50"     : round((hi_L.dur_p50  + hi_R.dur_p50)  / 2, 4),
        "hi_dur_p95"     : round((hi_L.dur_p95  + hi_R.dur_p95)  / 2, 4),
        "hi_dur_max"     : round(max(hi_L.dur_max, hi_R.dur_max), 4),
        # ---- delta ----
        "delta_n_steps"  : total_hi - total_lo,
        "pct_delta"      : round(100 * (total_hi - total_lo) / max(total_lo, 1), 2),
        "delta_n_long"   : (hi_L.n_long + hi_R.n_long) - (lo_L.n_long + lo_R.n_long),
    }


# ---------------------------------------------------------------------------
# Figures comparatives
# ---------------------------------------------------------------------------

def _shade(ax: plt.Axes, segs: list[tuple[int, int]], time: np.ndarray,
           color: str, alpha: float, label: str | None = None) -> None:
    for i, (s, e) in enumerate(segs):
        ax.axvspan(
            time[s], time[min(e, len(time) - 1)],
            color=color, alpha=alpha, linewidth=0,
            label=label if i == 0 else None,
        )


def _plot_comparison(
    sig: pd.DataFrame,
    subject_id: str,
    session: int,
    group: str,
    out_dir: Path,
) -> None:
    """
    Figure a 4 panneaux pour un sujet × session :
      - Ligne 1 L : signal + segments 0.05 (bleu) et 0.08 (orange) superposes
      - Ligne 1 R : idem pour le pied droit
      - Ligne 2 L : zoom 15s sur la zone avec le plus de differences
      - Ligne 2 R : idem pied droit
    """
    time   = sig["time"].values
    L_sig  = sig["total_L"].values
    R_sig  = sig["total_R"].values

    segs_lo_L = segment_steps(L_sig, _RATIO_LOW)
    segs_hi_L = segment_steps(L_sig, _RATIO_HIGH)
    segs_lo_R = segment_steps(R_sig, _RATIO_LOW)
    segs_hi_R = segment_steps(R_sig, _RATIO_HIGH)

    # Trouver la zone avec le plus de differences entre les deux seuils
    # Heuristique : chercher les pas dont la duree > 1.5s a 0.05 (candidats fusion)
    def _long_segs(segs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(s, e) for s, e in segs if (e - s) / _FS > 1.5]

    candidates = _long_segs(segs_lo_L) + _long_segs(segs_lo_R)
    if candidates:
        # Zoom centre sur le premier long segment
        s0, e0   = candidates[0]
        zoom_mid = (s0 + e0) / (2 * _FS)
        z_start  = max(0.0, zoom_mid - 7.5)
        z_end    = min(time[-1], zoom_mid + 7.5)
    else:
        # Pas de long segment : zoom sur le milieu de l'enregistrement
        z_start = max(0.0, time[-1] / 2 - 7.5)
        z_end   = min(time[-1], time[-1] / 2 + 7.5)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"Comparaison seuil stance : 0.05 vs 0.08\n"
        f"{subject_id}  |  Session {session}  |  Groupe {group}",
        fontsize=12, fontweight="bold",
    )

    # Legendes communes
    patch_lo = mpatches.Patch(color="steelblue", alpha=0.5,
                               label=f"0.05 — L:{len(segs_lo_L)} R:{len(segs_lo_R)}")
    patch_hi = mpatches.Patch(color="darkorange", alpha=0.5,
                               label=f"0.08 — L:{len(segs_hi_L)} R:{len(segs_hi_R)}")

    _config = [
        (axes[0, 0], L_sig, segs_lo_L, segs_hi_L, "steelblue",  "Pied gauche (L) — vue complete"),
        (axes[0, 1], R_sig, segs_lo_R, segs_hi_R, "seagreen",   "Pied droit (R) — vue complete"),
        (axes[1, 0], L_sig, segs_lo_L, segs_hi_L, "steelblue",  f"Pied gauche (L) — zoom {z_start:.0f}–{z_end:.0f}s"),
        (axes[1, 1], R_sig, segs_lo_R, segs_hi_R, "seagreen",   f"Pied droit (R) — zoom {z_start:.0f}–{z_end:.0f}s"),
    ]

    for i, (ax, fsig, slo, shi, sig_color, title) in enumerate(_config):
        ax.plot(time, fsig, color=sig_color, linewidth=0.7, alpha=0.85, zorder=3)
        _shade(ax, slo, time, color="steelblue",  alpha=0.25)
        _shade(ax, shi, time, color="darkorange", alpha=0.25)
        # Hachures pour les differences : segments presents dans lo mais pas hi
        lo_set = set(slo)
        hi_set = set(shi)
        for seg in lo_set - hi_set:
            s, e = seg
            ax.axvspan(time[s], time[min(e, len(time)-1)],
                       color="red", alpha=0.12, linewidth=0,
                       hatch="//", edgecolor="red", label="Fusion cassee par 0.08")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Force (u.a.)")
        if i >= 2:
            ax.set_xlabel("Temps (s)")
            ax.set_xlim(z_start, z_end)
        ax.legend(handles=[patch_lo, patch_hi], loc="upper right", fontsize=8)

    save_fig(fig, out_dir, f"thresh_cmp_{subject_id}_s{session:02d}")
    print(f"  Figure : thresh_cmp_{subject_id}_s{session:02d}.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{'='*60}")
    print(f"  COMPARAISON SEUIL STANCE  (0.05 vs 0.08)")
    print(f"  {n} enregistrements analyses")
    print(f"{'='*60}")

    print(f"\n--- Nombre total de pas ---")
    print(f"  Seuil 0.05 : {df.lo_n_steps.sum():6d}  (moy/enreg {df.lo_n_steps.mean():.1f})")
    print(f"  Seuil 0.08 : {df.hi_n_steps.sum():6d}  (moy/enreg {df.hi_n_steps.mean():.1f})")
    print(f"  Delta      : {df.delta_n_steps.sum():+6d}  "
          f"({df.pct_delta.mean():+.1f}% en moyenne par enreg)")

    print(f"\n--- Pas trop longs (> 2.0 s) ---")
    print(f"  Seuil 0.05 : {df.lo_n_long.sum():5d}")
    print(f"  Seuil 0.08 : {df.hi_n_long.sum():5d}")
    print(f"  Reduction  : {df.lo_n_long.sum() - df.hi_n_long.sum():+5d}")

    print(f"\n--- Pas trop courts (< 0.1 s) ---")
    print(f"  Seuil 0.05 : {df.lo_n_short.sum():5d}")
    print(f"  Seuil 0.08 : {df.hi_n_short.sum():5d}")

    print(f"\n--- Duree mediane des pas ---")
    print(f"  Seuil 0.05 : {df.lo_dur_p50.mean():.3f} s")
    print(f"  Seuil 0.08 : {df.hi_dur_p50.mean():.3f} s")

    print(f"\n--- Enregistrements avec forte variation (+/- 5 pas) ---")
    flagged = df[df.delta_n_steps.abs() >= 5].sort_values(
        "delta_n_steps", ascending=False
    )
    print(f"  {len(flagged)} enregistrements affectes")

    print(f"\n--- Top 10 gains de pas (0.08 casse des fusions) ---")
    top_gain = df.nlargest(10, "delta_n_steps")[
        ["subject_id", "session", "group", "lo_n_steps", "hi_n_steps",
         "delta_n_steps", "lo_n_long", "hi_n_long"]
    ]
    print(top_gain.to_string(index=False))

    print(f"\n--- Top 10 pertes de pas (0.08 sur-segmente ou fusionne) ---")
    top_loss = df.nsmallest(10, "delta_n_steps")[
        ["subject_id", "session", "group", "lo_n_steps", "hi_n_steps",
         "delta_n_steps", "lo_n_short", "hi_n_short"]
    ]
    print(top_loss.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sessions: list[str] | None = None) -> None:
    if sessions is None:
        sessions = ["01"]

    print(f"Analyse sessions : {sessions}")
    index = load_dataset_index()
    mask  = index["has_signal"] & index["session"].astype(str).str.zfill(2).isin(
        [s.zfill(2) for s in sessions]
    )
    subset = index[mask].copy()
    print(f"{len(subset)} enregistrements a traiter...")

    records  : list[dict]    = []
    sig_cache: dict[str, pd.DataFrame] = {}

    for _, row in subset.iterrows():
        fp = str(row["filepath"])
        if fp not in sig_cache:
            sig_cache[fp] = load_signal_file(row["filepath"])
        sig = sig_cache[fp]
        records.append(_compare_recording(row, sig))

    df = pd.DataFrame(records)

    # ---- Export CSV ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comp_path = OUTPUT_DIR / "stance_threshold_comparison.csv"
    df.to_csv(comp_path, index=False)
    print(f"CSV : {comp_path}")

    suspects = df[df.delta_n_steps.abs() >= 5].sort_values(
        "delta_n_steps", ascending=False
    )
    susp_path = OUTPUT_DIR / "stance_threshold_suspects.csv"
    suspects.to_csv(susp_path, index=False)
    print(f"CSV suspects : {susp_path}  ({len(suspects)} lignes)")

    # ---- Console ----
    _print_summary(df)

    # ---- Figures ----
    _OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Sujets focus + top 5 gains (fusions les plus marquees)
    top_subjects = suspects.nlargest(5, "delta_n_steps")["subject_id"].tolist()
    figure_targets = list(dict.fromkeys(_FOCUS_SUBJECTS + top_subjects))

    print(f"Figures pour : {figure_targets}")
    for subj in figure_targets:
        subj_rows = subset[subset["subject_id"] == subj]
        # Une seule session par sujet (session 01 si dispo, sinon la premiere)
        sess_rows = subj_rows[subj_rows["session"] == 1]
        if sess_rows.empty:
            sess_rows = subj_rows.head(1)
        for _, row in sess_rows.iterrows():
            fp = str(row["filepath"])
            if fp not in sig_cache:
                sig_cache[fp] = load_signal_file(row["filepath"])
            _plot_comparison(
                sig_cache[fp],
                subject_id=row["subject_id"],
                session=int(row["session"]),
                group=row["group"],
                out_dir=_OUT_FIG_DIR,
            )

    print(f"\nFigures dans : {_OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
