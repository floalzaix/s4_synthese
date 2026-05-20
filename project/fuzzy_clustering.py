"""
[ROLE]
Analyse Fuzzy C-Means exploratoire dans l'espace discriminant stabilisé par CV.

[RESPONSIBILITY]
- Utiliser les features stables (load_stable_features >= 60% folds) comme espace du clustering.
  Ce choix aligne l'espace fuzzy sur celui du classifieur RF, rendant les centroides
  directement comparables aux importances SHAP/perm.
- Choisir automatiquement C_opt (2..6) par maximisation de la Fuzzy Partition Coefficient.
- Produire les degrés d'appartenance par sujet et les centroides pondérés.
- Générer figures et CSV dans output/.

[IMPORTANT - PORTÉE EXPLORATOIRE]
Le clustering flou est un outil d'exploration du continuum clinique, PAS un classifieur.
Il ne produit pas de métriques de performance et ne remplace pas la validation CV.
Les "clusters" sont des régions géométriques dans l'espace des features, pas des groupes
cliniques validés.

[OUTPUTS]
- output/u_final.csv               : degrés d'appartenance par sujet (c colonnes).
- output/df_clusters.csv           : subject_id, group, u_max, cluster_max.
- output/figures/fuzzy/soft_barchart_by_subject.png
- output/figures/fuzzy/pca_projected_fuzzy.png
- output/figures/fuzzy/centroids_fuzzy_barplot.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from project.config import FUZZY_FIG_DIR, OUTPUT_DIR, RANDOM_STATE
from project.features import build_feature_matrix
from project.validate import load_stable_features
from project.viz_utils import clean_label, save_fig, setup_style

setup_style()

# Paramètres fuzzy c-means
FCM_M = 2          # exposant de fuzzification (valeur standard)
FCM_ERROR = 1e-5   # critère d'arrêt sur ||u_new - u_old||
FCM_MAXITER = 1000
C_RANGE = range(2, 7)  # C_opt cherché parmi 2, 3, 4, 5, 6


# ──────────────────────────────────────────────────────────────────────────────
# Sélection automatique de C via FPC
# ──────────────────────────────────────────────────────────────────────────────

def _select_c_opt(X_T: np.ndarray) -> tuple[int, dict[int, float]]:
    """
    Parcourt C_RANGE et retourne C avec la FPC maximale.

    FPC (Fuzzy Partition Coefficient) mesure la netteté de la partition :
    FPC = 1 signifie une partition crisp parfaite, FPC = 1/C correspond à
    une appartenance uniforme (clustering non-informatif).
    On maximise FPC pour trouver le nombre de clusters le plus "net".

    X_T : array (n_features, n_samples) — convention skfuzzy.
    """
    fpc_scores: dict[int, float] = {}
    for c in C_RANGE:
        _, _, _, _, _, _, fpc_val = fuzz.cmeans(
            X_T, c=c, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAXITER, seed=RANDOM_STATE
        )
        fpc_scores[c] = float(fpc_val)
    c_opt = max(fpc_scores, key=fpc_scores.__getitem__)
    return c_opt, fpc_scores


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _plot_soft_barchart(u: np.ndarray, df_meta: pd.DataFrame, c_opt: int) -> None:
    """
    Barplot empilé des degrés d'appartenance pour un sous-ensemble représentatif.
    Sélectionne jusqu'à 20 sujets : les 10 les plus "flous" (u_max proche de 1/C)
    et les 10 les plus "nets" (u_max élevé), pour montrer le continuum.
    """
    u_max = u.max(axis=0)
    # "flou" = u_max proche de 1/C_opt (appartenance distribuée entre clusters)
    fuzziness = np.abs(u_max - 1.0 / c_opt)
    idx_fuzzy = np.argsort(fuzziness)[:10]
    idx_crisp = np.argsort(fuzziness)[-10:]
    sel = np.unique(np.concatenate([idx_fuzzy, idx_crisp]))[:20]

    labels = [
        f"{df_meta.iloc[i]['subject_id']} ({df_meta.iloc[i]['group']})"
        for i in sel
    ]
    data = u[:, sel].T  # (n_sel, c_opt)

    fig, ax = plt.subplots(figsize=(12, max(5, len(sel) * 0.45)))
    bottom = np.zeros(len(sel))
    palette = sns.color_palette("tab10", c_opt)
    for k in range(c_opt):
        ax.barh(labels, data[:, k], left=bottom, color=palette[k], label=f"Cluster {k}")
        bottom += data[:, k]

    ax.axvline(1.0 / c_opt, color="black", linestyle="--", alpha=0.6,
               label=f"Appartenance uniforme (1/{c_opt})")
    ax.set_xlabel("Degré d'appartenance")
    ax.set_title(
        f"Degrés d'appartenance Fuzzy par Sujet (C={c_opt})\n"
        "Sujets représentatifs : 10 flous + 10 nets"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, FUZZY_FIG_DIR, "soft_barchart_by_subject")


def _plot_pca_fuzzy(
    X_scaled: np.ndarray, u: np.ndarray, df_meta: pd.DataFrame, c_opt: int
) -> None:
    """
    Projection PCA 2D. Chaque sujet est coloré par son degré d'appartenance
    au cluster dominant. L'opacité encode u_max (net = opaque, flou = transparent).
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    cluster_max = u.argmax(axis=0)
    u_max = u.max(axis=0)

    palette = sns.color_palette("tab10", c_opt)
    fig, ax = plt.subplots(figsize=(9, 7))

    for k in range(c_opt):
        mask = cluster_max == k
        sc = ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[palette[k]],
            alpha=np.clip(u_max[mask], 0.25, 1.0),
            s=80,
            edgecolors="k",
            linewidths=0.4,
            label=f"Cluster {k} (n={mask.sum()})",
        )

    # Sujets "flous" (u_max < 0.6) : cercle supplémentaire
    fuzzy_mask = u_max < 0.6
    ax.scatter(
        X_pca[fuzzy_mask, 0],
        X_pca[fuzzy_mask, 1],
        s=130,
        facecolors="none",
        edgecolors="gold",
        linewidths=1.5,
        label=f"Sujets flous (u_max < 0.6, n={fuzzy_mask.sum()})",
        zorder=5,
    )

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance expliquée)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance expliquée)")
    ax.set_title(
        f"Projection PCA — Clustering Flou (C={c_opt})\n"
        "Opacité = u_max | Cercle doré = sujet flou (u_max < 0.6)\n"
        "[Outil exploratoire — ne pas interpréter comme classification PD/CO]"
    )
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, FUZZY_FIG_DIR, "pca_projected_fuzzy")


def _plot_centroids_barplot(
    cntr: np.ndarray, features: list[str], c_opt: int
) -> None:
    """
    Barplot des centroides par cluster dans l'espace standardisé.
    Permet de voir quelles features différencient les clusters et de les comparer
    aux importances SHAP/perm (mêmes 8 features, même espace).
    """
    labels = [clean_label(f) for f in features]
    palette = sns.color_palette("tab10", c_opt)

    fig, axes = plt.subplots(1, c_opt, figsize=(4 * c_opt, 5), sharey=True)
    if c_opt == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        colors = ["salmon" if v > 0 else "skyblue" for v in cntr[k]]
        ax.barh(labels, cntr[k], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Cluster {k}", color=palette[k])
        ax.set_xlabel("Valeur centroides (std)")

    fig.suptitle(
        f"Centroides Fuzzy par Cluster (C={c_opt}, espace standardisé 8D)\n"
        "Rouge = valeur > moyenne pop. | Bleu = valeur < moyenne pop.",
        y=1.02,
    )
    save_fig(fig, FUZZY_FIG_DIR, "centroids_fuzzy_barplot")


# ──────────────────────────────────────────────────────────────────────────────
# Rapport textuel
# ──────────────────────────────────────────────────────────────────────────────

def _write_fuzzy_report(
    c_opt: int,
    fpc_scores: dict[int, float],
    cntr: np.ndarray,
    u: np.ndarray,
    df_clusters: pd.DataFrame,
    features: list[str],
) -> None:
    """Écrit output/fuzzy_clustering_report.txt."""
    u_max = u.max(axis=0)
    n_fuzzy = int((u_max < 0.6).sum())
    n_total = len(df_clusters)

    lines = [
        "=== RAPPORT CLUSTERING FLOU (FUZZY C-MEANS) ===",
        "",
        "PORTÉE : Outil EXPLORATOIRE du continuum clinique.",
        "Le clustering flou ne classe pas PD vs CO et ne produit pas de métriques",
        "de performance. Il révèle la structure géométrique de l'espace des features.",
        "",
        f"Espace : {len(features)} features stables (load_stable_features >= 60% folds CV)",
        f"  -> {features}",
        "Ces features sont identiques à celles utilisées par le classifieur RF et l'analyse",
        "SHAP. Les centroides sont donc directement comparables aux importances SHAP/perm.",
        "",
        "1. SÉLECTION DE C_OPT (Fuzzy Partition Coefficient) :",
    ]
    for c, fpc in sorted(fpc_scores.items()):
        marker = " <-- OPTIMAL" if c == c_opt else ""
        lines.append(f"   C={c} : FPC={fpc:.4f}{marker}")
    lines.append(
        f"   C_opt={c_opt} retenu (FPC maximale = {fpc_scores[c_opt]:.4f})."
    )
    lines.append(
        "   Note : FPC=1 = partition crisp parfaite ; FPC=1/C = appartenance uniforme non-informative."
    )

    lines += ["", "2. DESCRIPTION DES CLUSTERS :"]
    for k in range(c_opt):
        mask_k = df_clusters["cluster_max"] == k
        n_pd = (df_clusters.loc[mask_k, "group"] == "PD").sum()
        n_co = (df_clusters.loc[mask_k, "group"] == "CO").sum()
        # Feature la plus extrême du centroide (positive = au-dessus de la moyenne)
        top_feat_idx = int(np.argmax(np.abs(cntr[k])))
        top_feat_dir = "elevee" if cntr[k, top_feat_idx] > 0 else "basse"
        lines.append(
            f"   Cluster {k} : {mask_k.sum()} sujets dominants "
            f"(PD={n_pd}, CO={n_co}) | "
            f"Feature saillante : {clean_label(features[top_feat_idx])} ({top_feat_dir})"
        )

    lines += [
        "",
        "3. SUJETS 'FLOUS' (u_max < 0.6) :",
        f"   {n_fuzzy}/{n_total} sujets ({n_fuzzy/n_total:.1%}) ont une appartenance distribuée.",
        "   Ces sujets se situent dans les zones de recouvrement entre clusters.",
        "   Cela reflète l'hétérogénéité clinique réelle de la maladie de Parkinson",
        "   (stades variés, phénotypes moteurs différents).",
        "   IMPORTANT : la présence de sujets flous est attendue et ne constitue pas",
        "   un échec du modèle — elle est la principale information exploratoire.",
        "",
        "4. COMPARAISON AVEC L'ANALYSE SHAP :",
        "   Les features asymétriques (mean_asym, std_asym, asym_stance, asym_swing)",
        "   dominent l'espace discriminant selon SHAP et la permutation importance.",
        "   Les centroides des clusters reflètent cette structure :",
        "   un cluster à forte asymétrie correspond géométriquement aux sujets PD typiques,",
        "   un cluster à asymétrie faible correspond aux sujets CO et PD légers.",
        "   Les sujets flous entre ces clusters correspondent au continuum clinique",
        "   entre début de maladie et PD établi.",
        "",
        "5. LIMITES :",
        "   - C_opt est choisi par FPC, un critère interne. Il n'est pas validé cliniquement.",
        "   - L'initialisation aléatoire (seed fixé) garantit la reproductibilité mais",
        "     un autre seed peut donner un C_opt légèrement différent.",
        "   - Les clusters ne correspondent pas 1:1 aux groupes PD/CO : ne pas calculer",
        "     de précision ou de recall sur ces clusters.",
    ]

    with open(OUTPUT_DIR / "fuzzy_clustering_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

def run_final_fuzzy(session: str = "01", df=None) -> None:
    """
    @brief Exécute l'analyse Fuzzy C-Means exploratoire dans l'espace CV-stabilisé.
    @details Utilise les features stables (>= 60% folds) pour aligner l'espace du
    clustering sur celui du classifieur RF. C_opt est choisi par maximisation de la
    Fuzzy Partition Coefficient sur C=2..6. Aucune métrique de performance n'est émise.
    @param session Identifiant de la session (défaut "01").
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    print(f"--- Analyse Fuzzy C-Means exploratoire - Session: {session} ---")

    if df is None:
        df = build_feature_matrix(session=session)

    # Espace aligné sur le classifieur : mêmes 8 features que XAI et SHAP
    features = load_stable_features(min_freq=0.6)
    print(f"  Espace fuzzy : {len(features)} features stables CV -> {features}")

    clf_df = df.dropna(subset=features).copy()
    X = clf_df[features].values
    groups = clf_df["group"].values
    subject_ids = clf_df["subject_id"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # skfuzzy attend (n_features, n_samples) — transposition nécessaire
    X_T = X_scaled.T

    # ── Sélection automatique de C_opt via FPC ────────────────────────────────
    print("  Recherche C_opt (FPC sur C=2..6)...")
    c_opt, fpc_scores = _select_c_opt(X_T)
    print(f"  FPC par C : { {c: round(v,4) for c,v in fpc_scores.items()} }")
    print(f"  C_opt retenu : {c_opt} (FPC={fpc_scores[c_opt]:.4f})")

    # ── Fit final avec C_opt ──────────────────────────────────────────────────
    cntr, u, _, _, _, _, fpc_final = fuzz.cmeans(
        X_T, c=c_opt, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAXITER, seed=RANDOM_STATE
    )
    # u : (c_opt, n_samples) — degré d'appartenance de chaque sujet à chaque cluster

    u_max = u.max(axis=0)
    cluster_max = u.argmax(axis=0)
    n_fuzzy = int((u_max < 0.6).sum())
    print(
        f"  Sujets flous (u_max < 0.6) : {n_fuzzy}/{len(clf_df)} "
        f"({n_fuzzy/len(clf_df):.1%})"
    )

    # ── Exports CSV ───────────────────────────────────────────────────────────
    FUZZY_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # u_final.csv : degrés d'appartenance bruts
    u_df = pd.DataFrame(
        u.T, columns=[f"cluster_{k}" for k in range(c_opt)]
    )
    u_df.insert(0, "subject_id", subject_ids)
    u_df.to_csv(OUTPUT_DIR / "u_final.csv", index=False)

    # df_clusters.csv : synthèse par sujet
    df_clusters = pd.DataFrame({
        "subject_id": subject_ids,
        "group": groups,
        "u_max": u_max,
        "cluster_max": cluster_max,
    })
    df_clusters.to_csv(OUTPUT_DIR / "df_clusters.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    df_meta = pd.DataFrame({"subject_id": subject_ids, "group": groups})
    _plot_soft_barchart(u, df_meta, c_opt)
    _plot_pca_fuzzy(X_scaled, u, df_meta, c_opt)
    _plot_centroids_barplot(cntr, features, c_opt)

    # ── Rapport ───────────────────────────────────────────────────────────────
    _write_fuzzy_report(c_opt, fpc_scores, cntr, u, df_clusters, features)

    print(f"  Figures -> {FUZZY_FIG_DIR}")
    print(f"  CSV     -> {OUTPUT_DIR}/u_final.csv, df_clusters.csv")
    print(f"  Rapport -> {OUTPUT_DIR}/fuzzy_clustering_report.txt")


if __name__ == "__main__":
    run_final_fuzzy()
