"""
qc_long_strides.py — QC visuel des strides avec distance temporelle L/R > 100 samples.

Objectif : verifier que les paires (L, R) dont les midpoints sont eloignes de > 1s
correspondent a de vraies foulees longues ou a des erreurs de segmentation.

Produit une figure par sujet cle, avec :
  - Vue longue (signal complet) montrant tous les appuis detectes.
  - Vue zoomee sur les strides suspects (+/- 5s autour de la paire).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from project.config import REPO_ROOT, SEG_FIG_DIR
from project.load import load_signal_file
from project.viz_utils import FIG_WIDE, save_fig, setup_style

setup_style()

_FS = 100  # Hz
_DIST_THRESHOLD = 100  # samples (1.0 s)

# Sujets prioritaires issus de l'analyse QC (distribution + anomalies)
_TARGET_SUBJECTS = [
    "GaPt23",   # 27 strides suspects — le plus affecte du dataset
    "JuPt01",   # cas typique dur_L=0.98s / dur_R=2.42s
    "GaCo16",   # dur_L=2.69s / dur_R=4.19s — les deux jambes longues
    "GaPt31",   # dur_R=2.60s, plusieurs sessions
    "GaPt15",   # dur_L=0.14s — un pas gauche tres court dans la paire
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_signal(source_file: str) -> pd.DataFrame:
    return load_signal_file(REPO_ROOT / Path(source_file))


def _suspect_strides(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les paires (L, R) dont la distance entre midpoints est > _DIST_THRESHOLD.
    Chaque ligne = une paire, avec colonnes L_* et R_* + dist_samples.
    """
    paired = steps_df[steps_df.stride_id.notna()].copy()
    paired["stride_id"] = paired["stride_id"].astype(int)
    paired["midpoint"] = (paired.sample_start + paired.sample_end) / 2.0

    rows = []
    key_cols = ["step_id", "sample_start", "sample_end", "duration_s",
                "quality_flag", "midpoint"]
    for (subj, sess, sid), grp in paired.groupby(
        ["subject_id", "session", "stride_id"]
    ):
        if set(grp.foot.unique()) != {"L", "R"}:
            continue
        l = grp[grp.foot == "L"].iloc[0]
        r = grp[grp.foot == "R"].iloc[0]
        dist = abs(l.midpoint - r.midpoint)
        if dist > _DIST_THRESHOLD:
            row = {"subject_id": subj, "session": sess, "stride_id": sid,
                   "dist_samples": dist, "dist_s": round(dist / _FS, 3)}
            for col in key_cols:
                row[f"L_{col}"] = l[col]
                row[f"R_{col}"] = r[col]
            row["source_file"] = l.source_file
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _shade_steps(ax, signal: np.ndarray, steps_df: pd.DataFrame,
                 foot: str, color: str, time: np.ndarray,
                 suspect_step_ids: set[str]) -> None:
    """
    Trace le signal et colore chaque pas : vert = normal, rouge = suspect.
    """
    foot_steps = steps_df[steps_df.foot == foot]
    for _, row in foot_steps.iterrows():
        s, e = int(row.sample_start), int(row.sample_end)
        is_suspect = row.step_id in suspect_step_ids
        facecolor = "tomato" if is_suspect else color
        alpha = 0.35 if is_suspect else 0.15
        ax.axvspan(time[s], time[min(e, len(time) - 1)],
                   color=facecolor, alpha=alpha, linewidth=0)


def _mark_step(ax, signal: np.ndarray, s: int, e: int,
               time: np.ndarray, color: str, label_prefix: str) -> None:
    """Marque début/fin d'un pas avec des lignes verticales et des annotations."""
    t_s, t_e = time[s], time[min(e, len(time) - 1)]
    ax.axvline(t_s, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(t_e, color=color, linestyle=":",  linewidth=1.2, alpha=0.9)
    mid_t = (t_s + t_e) / 2
    y_pos = signal[s:e].max() * 1.05 if e > s else signal[s] * 1.05
    ax.annotate(
        f"{label_prefix}\n{t_e - t_s:.2f}s",
        xy=(mid_t, y_pos),
        ha="center", va="bottom", fontsize=7.5,
        color=color, fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Main figure per subject × session
# ---------------------------------------------------------------------------


def _plot_subject_session(
    subj: str,
    session: int,
    steps_df: pd.DataFrame,
    suspects: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Genere 1 figure a 3 panneaux pour un sujet × session :
      - Panneau 1 : signal complet L+R avec tous les appuis (rouge = suspect)
      - Panneau 2 : zoom autour du 1er stride suspect (+-5s)
      - Panneau 3 : zoom autour du 2e stride suspect si disponible, sinon resume
    """
    subset = steps_df[
        (steps_df.subject_id == subj) & (steps_df.session == session)
    ].copy()
    if subset.empty:
        return

    source_file = subset.iloc[0].source_file
    sig = _load_signal(source_file)
    time = sig["time"].values
    L_sig = sig["total_L"].values
    R_sig = sig["total_R"].values

    sess_suspects = suspects[
        (suspects.subject_id == subj) & (suspects.session == session)
    ].sort_values("dist_samples", ascending=False)

    if sess_suspects.empty:
        return

    # step_ids suspects pour coloration
    suspect_ids: set[str] = set(sess_suspects.L_step_id) | set(sess_suspects.R_step_id)

    # ---- Figure ----
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDE[0], 13))
    group = subset.iloc[0].get("group", "?")
    fig.suptitle(
        f"QC Long Strides — {subj}  |  Session {session}  |  Groupe {group}\n"
        f"{len(sess_suspects)} strides suspects (dist L/R > {_DIST_THRESHOLD/100:.1f}s)",
        fontsize=12, fontweight="bold",
    )

    # ---- Panneau 1 : vue longue ----
    ax = axes[0]
    offset = R_sig.max() * 1.1
    ax.plot(time, L_sig, color="steelblue", linewidth=0.8, alpha=0.9, label="L")
    ax.plot(time, R_sig + offset, color="seagreen",
            linewidth=0.8, alpha=0.9, label=f"R (offset +{offset:.0f})")
    _shade_steps(ax, L_sig, subset, "L", "steelblue", time, suspect_ids)
    _shade_steps(ax, R_sig + offset, subset, "R", "seagreen", time, suspect_ids)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel("Force (u.a.)")
    ax.set_title(f"Signal complet ({time[-1]:.0f}s) — rouge = pas implique dans un stride suspect")
    ax.legend(loc="upper right", fontsize=9)
    # Annotations des strides suspects sur la vue longue
    for _, sr in sess_suspects.iterrows():
        mid_t = (sr.L_midpoint + sr.R_midpoint) / (2 * _FS)
        ax.annotate(
            f"d={sr.dist_s:.2f}s", xy=(mid_t, L_sig.max() * 0.9),
            ha="center", fontsize=7, color="tomato", alpha=0.85,
        )

    # ---- Panneaux 2 et 3 : zooms sur les strides suspects ----
    for panel_idx, ax in enumerate(axes[1:], start=1):
        if panel_idx - 1 >= len(sess_suspects):
            ax.axis("off")
            ax.text(0.5, 0.5, "Pas d'autre stride suspect dans cette session",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            continue

        sr = sess_suspects.iloc[panel_idx - 1]
        # Fenetre : 5s avant le premier pas, 5s apres le dernier
        t_start = max(0, min(sr.L_sample_start, sr.R_sample_start) / _FS - 5.0)
        t_end   = min(time[-1], max(sr.L_sample_end, sr.R_sample_end) / _FS + 5.0)
        window = (time >= t_start) & (time <= t_end)

        ax.plot(time[window], L_sig[window],
                color="steelblue", linewidth=1.0, alpha=0.9, label="L")
        ax.plot(time[window], R_sig[window],
                color="seagreen",  linewidth=1.0, alpha=0.9, label="R")

        # Colorier tous les appuis dans la fenetre
        win_steps = subset[
            (subset.sample_start / _FS >= t_start) &
            (subset.sample_end   / _FS <= t_end)
        ]
        for _, row in win_steps[win_steps.foot == "L"].iterrows():
            s, e = int(row.sample_start), int(row.sample_end)
            is_s = row.step_id in suspect_ids
            ax.axvspan(time[s], time[min(e, len(time)-1)],
                       color="tomato" if is_s else "steelblue",
                       alpha=0.3 if is_s else 0.12, linewidth=0)
        for _, row in win_steps[win_steps.foot == "R"].iterrows():
            s, e = int(row.sample_start), int(row.sample_end)
            is_s = row.step_id in suspect_ids
            ax.axvspan(time[s], time[min(e, len(time)-1)],
                       color="tomato" if is_s else "seagreen",
                       alpha=0.3 if is_s else 0.12, linewidth=0)

        # Marqueurs precis sur le stride suspect
        _mark_step(ax, L_sig, int(sr.L_sample_start), int(sr.L_sample_end),
                   time, "steelblue", f"L ({sr.L_duration_s:.2f}s)")
        _mark_step(ax, R_sig, int(sr.R_sample_start), int(sr.R_sample_end),
                   time, "seagreen",  f"R ({sr.R_duration_s:.2f}s)")

        # Ligne reliant les midpoints
        mid_L = sr.L_midpoint / _FS
        mid_R = sr.R_midpoint / _FS
        y_arrow = max(L_sig[window].max(), R_sig[window].max()) * 1.02
        ax.annotate(
            "", xy=(mid_R, y_arrow), xytext=(mid_L, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="tomato", lw=1.5),
        )
        ax.text((mid_L + mid_R) / 2, y_arrow * 1.01,
                f"dist={sr.dist_s:.3f}s", ha="center", fontsize=9,
                color="tomato", fontweight="bold")

        ax.set_xlim(t_start, t_end)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Force (u.a.)")
        ax.set_title(
            f"Zoom stride #{int(sr.stride_id)} — dist L/R={sr.dist_s:.3f}s  |  "
            f"dur_L={sr.L_duration_s:.3f}s  dur_R={sr.R_duration_s:.3f}s"
        )
        legend_patches = [
            mpatches.Patch(color="steelblue", alpha=0.5, label="Appui L normal"),
            mpatches.Patch(color="seagreen",  alpha=0.5, label="Appui R normal"),
            mpatches.Patch(color="tomato",    alpha=0.5, label="Appui dans stride suspect"),
            plt.Line2D([0], [0], color="steelblue", ls="--", lw=1.2, label="Debut pas L"),
            plt.Line2D([0], [0], color="steelblue", ls=":",  lw=1.2, label="Fin pas L"),
            plt.Line2D([0], [0], color="seagreen",  ls="--", lw=1.2, label="Debut pas R"),
            plt.Line2D([0], [0], color="seagreen",  ls=":",  lw=1.2, label="Fin pas R"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7.5, ncol=2)

    save_fig(fig, out_dir, f"long_stride_check_{subj}_s{session:02d}")
    print(f"  Sauvegarde : long_stride_check_{subj}_s{session:02d}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Genere les figures QC pour les strides suspects des sujets cibles.
    Produit aussi un tableau CSV de synthese.
    """
    out_dir = SEG_FIG_DIR

    print("Chargement de steps.csv...")
    from project.config import OUTPUT_DIR
    steps_path = OUTPUT_DIR / "steps.csv"
    if not steps_path.exists():
        raise FileNotFoundError(
            f"steps.csv introuvable : {steps_path}\n"
            "Executer d'abord : python -m project.build_steps_index"
        )
    steps_df = pd.read_csv(steps_path)

    print("Calcul des strides suspects (dist L/R > 100 samples)...")
    suspects = _suspect_strides(steps_df)
    print(f"  -> {len(suspects)} strides suspects sur "
          f"{suspects.subject_id.nunique()} sujets")

    # Sauvegarde du tableau de synthese
    suspects_path = OUTPUT_DIR / "long_stride_suspects.csv"
    suspects.to_csv(suspects_path, index=False)
    print(f"  Tableau CSV : {suspects_path}")

    # Sujets a tracer : les cibles prioritaires + les 5 sujets les plus affectes
    top5 = (
        suspects.groupby("subject_id")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    targets = list(dict.fromkeys(_TARGET_SUBJECTS + top5))  # dedoublonne, ordre stable

    print(f"\nGeneration des figures pour : {targets}")
    for subj in targets:
        subj_suspects = suspects[suspects.subject_id == subj]
        sessions = sorted(subj_suspects.session.unique())
        # Traiter au plus 2 sessions par sujet pour garder le rapport lisible
        for sess in sessions[:2]:
            _plot_subject_session(subj, sess, steps_df, suspects, out_dir)

    print(f"\nDone. Figures dans : {out_dir}")


if __name__ == "__main__":
    main()
