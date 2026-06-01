"""
[ROLE]
QC visuel des pas identifiés comme "pd_mismatch" ou "high_asym" par model_step_comparison.py.

[RESPONSIBILITY]
- Charger focus_malades_premiers_pas.csv et l'enrichir avec source_file/sample_start/sample_end
  depuis steps.csv (jointure sur step_id).
- Pour chaque pas sélectionné : générer une figure 2 panneaux (contexte + zoom).
- Produire un résumé statistique visuel (distributions, barplot, scatter).
- Sauvegarder un CSV de synthèse par anomaly_type.

[OUTPUTS]
- output/figures/step_qc/step_qc_{anomaly_type}_{step_id}.png   (une par pas)
- output/figures/step_qc/step_qc_summary.png                    (résumé 4 sous-graphiques)
- output/etude_du_pas/step_qc_summary.csv

[DEPENDENCIES]
- gaitpdb.config, gaitpdb.load, gaitpdb.viz.utils
- output/steps.csv
- output/etude_du_pas/focus_malades_premiers_pas.csv  (généré par model_step_comparison.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gaitpdb.config import OUTPUT_DIR, REPO_ROOT, STEP_QC_FIG_DIR
from gaitpdb.load import load_signal_file
from gaitpdb.viz.utils import FIG_LARGE, FIG_WIDE, save_fig, setup_style

setup_style()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_STEP_OUT_DIR: Path = OUTPUT_DIR / "etude_du_pas"
_FS: int = 100                # Hz
_CONTEXT_S: float = 4.0       # secondes de contexte autour du pas dans le panneau 1
_MAX_FIGS_PER_TYPE: int = 20  # cap de figures par type : évite des milliers de PNG

_ANOMALY_PALETTE: dict[str, str] = {
    "pd_mismatch": "tomato",
    "high_asym":   "darkorange",
    "both":        "purple",
}


# ---------------------------------------------------------------------------
# Chargement et enrichissement
# ---------------------------------------------------------------------------


def _load_enriched_focus(
    focus_path: Path,
    steps_csv_path: Path,
) -> pd.DataFrame:
    """
    Charge focus_malades_premiers_pas.csv et joint steps.csv pour récupérer
    source_file, sample_start, sample_end (absents du focus d'origine).
    """
    focus = pd.read_csv(focus_path)
    steps = pd.read_csv(steps_csv_path, usecols=[
        "step_id", "source_file", "sample_start", "sample_end",
    ])
    merged = focus.merge(steps, on="step_id", how="left")
    # Garder uniquement les types d'anomalie traités ici
    merged = merged[merged["anomaly_type"].isin({"pd_mismatch", "high_asym", "both"})]
    return merged.reset_index(drop=True)


def _select_top_steps(focus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les pas les plus pertinents à tracer pour chaque type.

    Stratégie :
      pd_mismatch : top _MAX_FIGS_PER_TYPE par y_prob DESC — les prédictions les plus
                    confiantes. Ces cas sont les plus informatifs cliniquement.
      high_asym   : top _MAX_FIGS_PER_TYPE par stride_asym_peak DESC (fallback sur
                    stride_asym_duration si NaN) — les asymétries les plus extrêmes.
      both        : tous conservés — ce cumul est rare et toujours pertinent à inspecter.
    """
    parts: list[pd.DataFrame] = []

    mask_mm = focus_df["anomaly_type"] == "pd_mismatch"
    if mask_mm.any():
        parts.append(
            focus_df[mask_mm].nlargest(_MAX_FIGS_PER_TYPE, "y_prob")
        )

    mask_ha = focus_df["anomaly_type"] == "high_asym"
    if mask_ha.any():
        sort_col = (
            "stride_asym_peak"
            if focus_df.loc[mask_ha, "stride_asym_peak"].notna().any()
            else "stride_asym_duration"
        )
        parts.append(
            focus_df[mask_ha].nlargest(_MAX_FIGS_PER_TYPE, sort_col)
        )

    mask_both = focus_df["anomaly_type"] == "both"
    if mask_both.any():
        parts.append(focus_df[mask_both])

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Chargement des signaux (avec cache par fichier source)
# ---------------------------------------------------------------------------


def _get_or_load(
    cache: dict[str, pd.DataFrame],
    source_file: str,
) -> pd.DataFrame:
    """Charge un signal source une seule fois et le met en cache."""
    if source_file not in cache:
        cache[source_file] = load_signal_file(REPO_ROOT / Path(source_file))
    return cache[source_file]


# ---------------------------------------------------------------------------
# Génération d'une figure QC par pas
# ---------------------------------------------------------------------------


def _plot_step_qc(
    row: pd.Series,
    sig: pd.DataFrame,
    partner_row: Optional[pd.Series] = None,
) -> plt.Figure:
    """
    Génère une figure à 2 panneaux pour un pas donné.

    Panneau 1 — Vue contextuelle ±_CONTEXT_S autour du pas :
      - total_L (bleu) et total_R (vert)
      - Fond coloré sur le pas cible
      - Fond gris sur le pas partenaire (si stride_id apparié)
      - Annotation de la prédiction RF et du type d'anomalie

    Panneau 2 — Zoom sur le pas lui-même :
      - total_L et total_R tronqués sur [sample_start:sample_end]
      - Ligne horizontale au niveau de peak_force
      - Textbox de synthèse des métriques clés
    """
    time = sig["time"].values
    L = sig["total_L"].values
    R = sig["total_R"].values

    start = int(row["sample_start"])
    end   = int(row["sample_end"])

    t_step_start = time[start]
    t_step_end   = time[min(end, len(time) - 1)]

    anomaly_color = _ANOMALY_PALETTE.get(str(row["anomaly_type"]), "tomato")

    fig, axes = plt.subplots(2, 1, figsize=FIG_WIDE)

    group_label = str(row.get("group", "?"))
    fig.suptitle(
        f"{row['step_id']}  |  anomalie: {row['anomaly_type']}  |  "
        f"Sujet {row['subject_id']} ({group_label})  |  pied {row['foot']}",
        fontsize=12, fontweight="bold",
    )

    # ── Panneau 1 : contexte ────────────────────────────────────────────
    ax1 = axes[0]
    t_ctx_start = max(time[0],  t_step_start - _CONTEXT_S)
    t_ctx_end   = min(time[-1], t_step_end   + _CONTEXT_S)
    win = (time >= t_ctx_start) & (time <= t_ctx_end)

    ax1.plot(time[win], L[win], color="steelblue",  linewidth=0.9, alpha=0.9, label="total_L")
    ax1.plot(time[win], R[win], color="seagreen",   linewidth=0.9, alpha=0.9, label="total_R")

    # Fond sur le pas cible
    ax1.axvspan(t_step_start, t_step_end, color=anomaly_color, alpha=0.25,
                linewidth=0, label="Pas cible")

    # Fond gris sur le pas partenaire (autre pied, même stride_id)
    if partner_row is not None:
        p_start = int(partner_row["sample_start"])
        p_end   = int(partner_row["sample_end"])
        ax1.axvspan(
            time[p_start], time[min(p_end, len(time) - 1)],
            color="gray", alpha=0.15, linewidth=0, label=f"Partenaire ({partner_row['foot']})",
        )

    # Annotation prédiction RF
    y_top = max(L[win].max(), R[win].max()) * 0.95
    ax1.text(
        t_step_start + (t_step_end - t_step_start) / 2, y_top,
        f"y_prob={row['y_prob']:.2f}",
        ha="center", va="top", fontsize=9,
        color=anomaly_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=anomaly_color),
    )

    ax1.set_xlim(t_ctx_start, t_ctx_end)
    ax1.set_ylabel("Force (u.a.)")
    ax1.set_title(f"Contexte ±{_CONTEXT_S:.0f}s — prédiction RF PD : y_prob={row['y_prob']:.3f}")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panneau 2 : zoom sur le pas ─────────────────────────────────────
    ax2 = axes[1]
    step_L = L[start:end]
    step_R = R[start:end]
    t_step = time[start:end]

    ax2.plot(t_step, step_L, color="steelblue", linewidth=1.2, label="total_L")
    ax2.plot(t_step, step_R, color="seagreen",  linewidth=1.2, label="total_R")

    # Ligne horizontale au niveau du pic de force
    peak = float(row.get("peak_force", max(step_L.max(), step_R.max())))
    ax2.axhline(peak, color=anomaly_color, linestyle="--", linewidth=1.0, alpha=0.7,
                label=f"peak_force = {peak:.0f}")

    # Marqueurs début / fin
    ax2.axvline(t_step_start, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.axvline(t_step_end,   color="gray", linestyle=":",  linewidth=1.0, alpha=0.8)

    # Textbox métriques
    asym_peak = row.get("stride_asym_peak", float("nan"))
    asym_dur  = row.get("stride_asym_duration", float("nan"))
    dur_s     = float(row.get("duration_s", (end - start) / _FS))

    def _fmt(v) -> str:
        return f"{float(v):.3f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "n/a"

    metrics_txt = (
        f"dur = {dur_s:.2f} s\n"
        f"peak = {peak:.0f}\n"
        f"asym_peak = {_fmt(asym_peak)}\n"
        f"asym_dur  = {_fmt(asym_dur)}\n"
        f"y_prob    = {row['y_prob']:.3f}"
    )
    ax2.text(
        0.99, 0.97, metrics_txt,
        transform=ax2.transAxes,
        ha="right", va="top", fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  alpha=0.85, edgecolor="gray"),
    )

    ax2.set_xlim(t_step_start, t_step_end)
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Force (u.a.)")
    ax2.set_title(
        f"Zoom sur le pas  |  qualité: {row.get('quality_flag', '?')}  |  "
        f"durée: {dur_s:.2f} s"
    )
    ax2.legend(loc="upper left", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Résumé statistique
# ---------------------------------------------------------------------------


def _generate_summary_figure(focus_df: pd.DataFrame) -> None:
    """
    Figure 4 panneaux résumant les distributions des métriques clés par anomaly_type.
    Sauvée dans STEP_QC_FIG_DIR/step_qc_summary.png.
    """
    palette = _ANOMALY_PALETTE
    fig, axes = plt.subplots(2, 2, figsize=FIG_LARGE)
    fig.suptitle("Résumé QC — Pas anomaliques (pd_mismatch / high_asym / both)",
                 fontsize=13, fontweight="bold")

    # 1. KDE y_prob par anomaly_type
    ax = axes[0, 0]
    for atype, color in palette.items():
        sub = focus_df[focus_df["anomaly_type"] == atype]["y_prob"].dropna()
        if len(sub) > 1:
            sns.kdeplot(sub, ax=ax, color=color, label=f"{atype} (n={len(sub)})", fill=True, alpha=0.3)
    ax.set_xlabel("y_prob (probabilité PD)")
    ax.set_title("Distribution de y_prob par type")
    ax.legend(fontsize=8)

    # 2. KDE stride_asym_peak par anomaly_type
    ax = axes[0, 1]
    for atype, color in palette.items():
        sub = focus_df[focus_df["anomaly_type"] == atype]["stride_asym_peak"].dropna()
        if len(sub) > 1:
            sns.kdeplot(sub, ax=ax, color=color, label=f"{atype} (n={len(sub)})", fill=True, alpha=0.3)
    ax.set_xlabel("stride_asym_peak")
    ax.set_title("Distribution de l'asymétrie de force")
    ax.legend(fontsize=8)

    # 3. Barplot count par anomaly_type × group
    ax = axes[1, 0]
    count_df = (
        focus_df.groupby(["anomaly_type", "group"])
        .size()
        .reset_index(name="count")
    )
    sns.barplot(
        data=count_df, x="anomaly_type", y="count", hue="group",
        palette={"PD": "salmon", "CO": "skyblue"},
        ax=ax,
    )
    ax.set_title("Nombre de pas par type et groupe")
    ax.set_xlabel("")
    ax.legend(title="Groupe", fontsize=8)

    # 4. Scatter y_prob vs stride_asym_peak
    ax = axes[1, 1]
    for atype, color in palette.items():
        sub = focus_df[focus_df["anomaly_type"] == atype]
        if sub.empty:
            continue
        x = sub["stride_asym_peak"].fillna(0)
        y = sub["y_prob"]
        ax.scatter(x, y, c=color, alpha=0.4, s=20, label=atype)
    ax.set_xlabel("stride_asym_peak")
    ax.set_ylabel("y_prob")
    ax.set_title("y_prob vs asymétrie de force")
    ax.legend(fontsize=8)

    save_fig(fig, STEP_QC_FIG_DIR, "step_qc_summary")


def _write_summary_csv(focus_df: pd.DataFrame, out_path: Path) -> None:
    """
    Exporte des statistiques agrégées par anomaly_type :
    count, mean/std de y_prob / stride_asym_peak / stride_asym_duration / duration_s,
    répartition PD/CO.
    """
    num_cols = ["y_prob", "stride_asym_peak", "stride_asym_duration", "duration_s"]
    num_cols = [c for c in num_cols if c in focus_df.columns]

    agg = focus_df.groupby("anomaly_type")[num_cols].agg(["count", "mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]

    # Répartition PD/CO
    group_counts = (
        focus_df.groupby(["anomaly_type", "group"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("n_")
    )
    summary = agg.join(group_counts)
    summary.to_csv(out_path)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Génère le QC visuel complet pour les pas pd_mismatch et high_asym.

    Prérequis : exécuter model_step_comparison.run_step_comparison() au préalable
    pour générer output/etude_du_pas/focus_malades_premiers_pas.csv.
    """
    focus_path = _STEP_OUT_DIR / "focus_malades_premiers_pas.csv"
    if not focus_path.exists():
        print(
            f"[step_qc] {focus_path} introuvable.\n"
            "  -> Exécuter d'abord : python -m gaitpdb.model_step_comparison"
        )
        return

    print(f"--- QC Visuel Pas Anomaliques ---")
    STEP_QC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Enrichissement : jointure avec steps.csv pour source_file/sample_start/end
    focus_df = _load_enriched_focus(focus_path, OUTPUT_DIR / "steps.csv")
    print(f"  Pas chargés : {len(focus_df)} anomalies")
    print(f"  Types : {dict(focus_df['anomaly_type'].value_counts())}")

    # Sélection des cas à tracer
    top_df = _select_top_steps(focus_df)
    print(f"  Sélectionnés pour tracé : {len(top_df)} pas "
          f"(max {_MAX_FIGS_PER_TYPE}/type sauf 'both')")

    # Chargement des signaux avec cache
    sig_cache: dict[str, pd.DataFrame] = {}

    # Pré-indexation steps.csv pour trouver les partenaires de foulée
    all_steps = pd.read_csv(OUTPUT_DIR / "steps.csv")

    for _, row in top_df.iterrows():
        sig = _get_or_load(sig_cache, str(row["source_file"]))

        # Trouver le pas partenaire (autre pied, même stride_id)
        partner_row = None
        if pd.notna(row.get("stride_id")):
            other_foot = "R" if row["foot"] == "L" else "L"
            partner_mask = (
                (all_steps["subject_id"] == row["subject_id"]) &
                (all_steps["session"].astype(str) == str(row["session"])) &
                (all_steps["stride_id"] == row["stride_id"]) &
                (all_steps["foot"] == other_foot)
            )
            matches = all_steps[partner_mask]
            if not matches.empty:
                partner_row = matches.iloc[0]

        fig = _plot_step_qc(row, sig, partner_row)
        fname = f"step_qc_{row['anomaly_type']}_{row['step_id']}"
        save_fig(fig, STEP_QC_FIG_DIR, fname)

    n_saved = len(top_df)
    print(f"  {n_saved} figures sauvegardées -> {STEP_QC_FIG_DIR}")

    # Résumé statistique
    _generate_summary_figure(focus_df)
    print(f"  Résumé visuel -> {STEP_QC_FIG_DIR}/step_qc_summary.png")

    _write_summary_csv(focus_df, _STEP_OUT_DIR / "step_qc_summary.csv")
    print(f"  Résumé CSV -> {_STEP_OUT_DIR}/step_qc_summary.csv")


if __name__ == "__main__":
    main()
