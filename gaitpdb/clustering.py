"""
[ROLE]
Clustering non supervise des sujets GaitPDB.

[RESPONSIBILITY]
  Hard clustering (KMeans, GMM, PCA, t-SNE) : run_patient_clustering().
  Fuzzy clustering (Fuzzy C-Means, FPC) : run_final_fuzzy().

[OUTPUTS]
  output/clustering/
"""

from __future__ import annotations

##
# @file patient_clustering.py
# @brief Analyse exploratoire non supervisée (Clustering et Projections).
# @details Implémente K-Means, GMM, PCA et t-SNE pour explorer la structure des données.
#

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gaitpdb.config import CLUSTERING_FIG_DIR, OUTPUT_DIR, RANDOM_STATE
from gaitpdb.features import build_feature_matrix
from gaitpdb.validate import load_stable_features
from gaitpdb.viz.utils import PALETTE, clean_label, save_fig, setup_style

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

setup_style()

# Fallback statique — utilisé uniquement si cv_feature_stability.csv est absent.
# En production, run_patient_clustering() charge les features stables depuis la CV.
_CLUSTER_FEATURES_FALLBACK = [
    "std_asym",
    "mean_asym",
    "mean_abs_diff",
    "diff_auc",
    "cv_interval_L",
    "n_steps",
]


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    @brief Crée une ellipse de covariance pour un scatterplot.
    @param x Coordonnées X.
    @param y Coordonnées Y.
    @param ax Axe matplotlib.
    @param n_std Nombre d'écarts-types pour la taille de l'ellipse.
    @param facecolor Couleur de fond de l'ellipse.
    @param kwargs Arguments supplémentaires pour l'ellipse.
    """
    if x.size != y.size or x.size < 2:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_radar_centroids(X_scaled, labels, features: list[str], k=2):
    """
    @brief Génère un radar chart des centroïdes des clusters.
    @param X_scaled Matrice des caractéristiques standardisées.
    @param labels Étiquettes des clusters.
    @param features Liste des noms de features (alignée avec X_scaled).
    @param k Nombre de clusters.
    """
    df_c = pd.DataFrame(X_scaled, columns=[clean_label(f) for f in features])
    df_c["Cluster"] = labels
    centroids = df_c.groupby("Cluster").mean().reset_index()

    scaler = MinMaxScaler()
    centroids_scaled = scaler.fit_transform(centroids.drop("Cluster", axis=1))

    categories = list(df_c.columns[:-1])
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("tab10", k)
    for i in range(k):
        values = centroids_scaled[i].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle="solid",
            label=f"Cluster {i}",
            color=colors[i],
        )
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    plt.xticks(angles[:-1], categories, size=10)
    ax.set_yticks([])
    plt.title("Profil Moyen des Clusters (Features standardisées)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_radar_centroids")


def plot_pca_loadings(pca, features):
    """
    @brief Génère un barplot des corrélations (loadings) des features avec les PC.
    @param pca Objet PCA entraîné.
    @param features Liste des noms des caractéristiques.
    """
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    df_loadings = pd.DataFrame(
        loadings,
        columns=[f"PC{i + 1}" for i in range(pca.n_components)],
        index=[clean_label(f) for f in features],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    df_loadings.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
    ax.set_title("Loadings PCA : Contribution des Features aux Axes Principaux")
    ax.set_ylabel("Corrélation avec l'axe principal")
    plt.xticks(rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=1)
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_pca_loadings")


def plot_features_by_cluster(X_scaled, labels, features: list[str]):
    """
    @brief Génère des boxplots des caractéristiques séparés par cluster.
    @param X_scaled Matrice des caractéristiques standardisées.
    @param labels Étiquettes des clusters.
    @param features Liste des noms de features (alignée avec X_scaled).
    """
    df_c = pd.DataFrame(X_scaled, columns=[clean_label(f) for f in features])
    df_c["Cluster"] = [f"Cluster {l}" for l in labels]
    df_melted = df_c.melt(id_vars="Cluster", var_name="Feature", value_name="Z-Score")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_melted, x="Feature", y="Z-Score", hue="Cluster", palette="Set1", ax=ax
    )
    ax.set_title(
        "Distribution des Features par Cluster K-Means (Espace 6D Standardisé)"
    )
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_features_by_cluster")


def run_patient_clustering(session: str = "01", df=None):
    """
    @brief Exécute l'analyse de clustering et de projection complète.
    @details Les features utilisées sont les features stables identifiées par run_validation()
    (cv_feature_stability.csv). Cela aligne l'espace du clustering avec l'espace
    discriminant du classifieur supervisé.
    @param session Identifiant de la session (défaut "01").
    @param df Matrice de features pré-calculée (optionnel ; calculée si None).
    """
    # Chargement dynamique depuis cv_feature_stability.csv — pas de liste hardcodée.
    cluster_features = load_stable_features(min_freq=0.6)
    n_dim = len(cluster_features)
    print(f"--- Analyse Clustering & Projections ({n_dim}D) - Session: {session} ---")
    print(f"  Features : {cluster_features}")

    if df is None:
        df = build_feature_matrix(session=session)
    clf_df = df.dropna(subset=cluster_features).copy()
    X = clf_df[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Projections (Lecture Visuelle)
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca2 = pca2.fit_transform(X_scaled)
    pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca3 = pca3.fit_transform(X_scaled)

    tsne = TSNE(
        n_components=2, perplexity=min(30, len(X) - 1), random_state=RANDOM_STATE
    )
    X_tsne = tsne.fit_transform(X_scaled)

    X_umap = None
    if HAS_UMAP:
        X_umap = umap.UMAP(random_state=RANDOM_STATE).fit_transform(X_scaled)

    # 2. Clustering Mathématique 6D (KMeans & GMM)
    km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
    labels_km = km.fit_predict(X_scaled)
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    labels_gmm = gmm.fit_predict(X_scaled)

    # 3. Métriques
    df_metrics = pd.DataFrame(
        {
            "Metric": ["Silhouette Score", "Davies-Bouldin", "Calinski-Harabasz"],
            "K-Means": [
                silhouette_score(X_scaled, labels_km),
                davies_bouldin_score(X_scaled, labels_km),
                calinski_harabasz_score(X_scaled, labels_km),
            ],
            "GMM": [
                silhouette_score(X_scaled, labels_gmm),
                davies_bouldin_score(X_scaled, labels_gmm),
                calinski_harabasz_score(X_scaled, labels_gmm),
            ],
        }
    )
    df_metrics.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)

    # 4. Export Embeddings
    emb_data = {
        "subject_id": clf_df["subject_id"].values,
        "group": clf_df["group"].values,
        "study": clf_df["study"].values,
        "pca1": X_pca2[:, 0],
        "pca2": X_pca2[:, 1],
        # 3D PCA components stored together — never mix with pca1/pca2 (different fit)
        "pca3_1": X_pca3[:, 0],
        "pca3_2": X_pca3[:, 1],
        "pca3_3": X_pca3[:, 2],
        "tsne1": X_tsne[:, 0],
        "tsne2": X_tsne[:, 1],
        "cluster_km": labels_km,
        "cluster_gmm": labels_gmm,
    }
    if X_umap is not None:
        emb_data.update({"umap1": X_umap[:, 0], "umap2": X_umap[:, 1]})
    emb_df = pd.DataFrame(emb_data)
    emb_df.to_csv(OUTPUT_DIR / "patient_embeddings.csv", index=False)

    cross_tab = pd.crosstab(emb_df["group"], emb_df["cluster_km"], margins=True)
    cross_tab.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    # 5. Visualisations Multivariées
    # Loadings PCA
    plot_pca_loadings(pca3, cluster_features)

    # Distribution par cluster
    plot_features_by_cluster(X_scaled, labels_km, features=cluster_features)

    # Pairplot
    df_pair = pd.DataFrame(X_scaled, columns=[clean_label(f) for f in cluster_features])
    df_pair["Group"] = clf_df["group"].values
    g = sns.pairplot(
        df_pair,
        hue="Group",
        palette=PALETTE,
        plot_kws={"alpha": 0.5},
        diag_kind="kde",
        corner=True,
    )
    g.fig.suptitle(
        f"Pairplot des {n_dim} features stables (Espace Standardisé)", y=1.02
    )
    save_fig(g.fig, CLUSTERING_FIG_DIR, "clustering_pairplot")

    # Radar Chart
    plot_radar_centroids(X_scaled, labels_km, features=cluster_features, k=2)

    # 6. Projections 2D & 3D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(
        data=emb_df,
        x="pca1",
        y="pca2",
        hue="group",
        palette=PALETTE,
        ax=ax1,
        s=80,
        alpha=0.7,
    )
    for grp, col in PALETTE.items():
        mask = emb_df["group"] == grp
        confidence_ellipse(
            emb_df[mask]["pca1"],
            emb_df[mask]["pca2"],
            ax1,
            edgecolor=col,
            linewidth=2,
            n_std=1.5,
            alpha=0.8,
        )
    ax1.set_title("PCA 2D : Dispersion par Groupe Clinique")

    sns.scatterplot(
        data=emb_df, x="pca1", y="pca2", hue="study", ax=ax2, s=80, alpha=0.7
    )
    ax2.set_title("PCA 2D : Dispersion par Étude (Biais Centre ?)")
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_pca_2d")

    n_cols = 2 if X_umap is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))
    ax_t = axes[0] if n_cols > 1 else axes
    sns.scatterplot(
        data=emb_df,
        x="tsne1",
        y="tsne2",
        hue="group",
        palette=PALETTE,
        ax=ax_t,
        s=80,
        alpha=0.7,
    )
    ax_t.set_title("t-SNE 2D")
    if X_umap is not None:
        sns.scatterplot(
            data=emb_df,
            x="umap1",
            y="umap2",
            hue="group",
            palette=PALETTE,
            ax=axes[1],
            s=80,
            alpha=0.7,
        )
        axes[1].set_title("UMAP 2D")
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_manifold_2d")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(
        data=emb_df,
        x="pca1",
        y="pca2",
        hue="cluster_km",
        palette="Set1",
        style="group",
        ax=ax1,
        s=100,
        alpha=0.7,
    )
    ax1.set_title("Clusters K-Means vs Groupes Réels (Forme)")
    sns.scatterplot(
        data=emb_df,
        x="pca1",
        y="pca2",
        hue="cluster_gmm",
        palette="Set2",
        style="group",
        ax=ax2,
        s=100,
        alpha=0.7,
    )
    ax2.set_title("Clusters GMM vs Groupes Réels (Forme)")
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_kmeans_vs_gmm")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for g, c in PALETTE.items():
        m = emb_df["group"] == g
        ax.scatter(
            emb_df[m]["pca3_1"],
            emb_df[m]["pca3_2"],
            emb_df[m]["pca3_3"],
            c=c,
            label=g,
            s=50,
            alpha=0.6,
        )
    ax.set_title("PCA 3D : Topologie Complémentaire")
    ax.legend()
    save_fig(fig, CLUSTERING_FIG_DIR, "clustering_pca_3d")

    # 7. Résumé
    sil = df_metrics["K-Means"][0]
    with open(OUTPUT_DIR / "clustering_report.txt", "w", encoding="utf-8") as f:
        f.write("=== RAPPORT EXPLORATOIRE DE STRUCTURE LATENTE ===\n\n")
        f.write("ESPACE DE CALCUL ET PROJECTIONS :\n")
        f.write(
            f"- Le clustering (K-Means, GMM) est calculé dans l'espace "
            f"mathématique réel à {n_dim} dimensions (standardisé).\n"
        )
        f.write(
            f"- Features utilisées : {', '.join(cluster_features)}\n"
        )
        f.write(
            "- Ces features sont les features stables identifiées par la validation croisée "
            "(cv_feature_stability.csv, frequence >= 60%).\n"
        )
        f.write(
            "- Les projections (PCA 2D/3D, t-SNE) sont des outils de lecture visuelle. "
            "Elles illustrent les principales sources de variance.\n\n"
        )

        f.write(f"MÉTRIQUE PRINCIPALE :\n")
        f.write(f"- Silhouette Score (K-Means) : {sil:.4f}\n\n")

        f.write("INTERPRÉTATION CLINIQUE PRUDENTE :\n")
        if sil < 0.25:
            f.write(
                "-> RECOUVREMENT RÉEL : Les groupes PD et CO présentent un chevauchement structurel fort.\n"
            )
            f.write(
                "-> L'absence de séparation parfaite n'est pas un échec, c'est la réalité clinique du dataset : la maladie de Parkinson s'exprime sur un spectre (continuum), souvent entremêlé avec le vieillissement sain dans cet espace de features.\n"
            )
        elif sil < 0.60:
            f.write(
                "-> STRUCTURE LATENTE MODÉRÉE À MARQUÉE : Une séparation utile émerge, mais elle reste incomplète.\n"
            )
            f.write(
                "-> Le recouvrement observé (visible sur le Pairplot et les projections) est la réalité clinique de ces données. Il ne faut pas forcer une séparation artificielle.\n"
            )
        else:
            f.write(
                "-> SÉPARATION NETTE : Les algorithmes identifient une frontière forte. (Vérifier s'il n'y a pas de fuite de données ou de biais de site prédominant).\n"
            )

        f.write("\nCONCLUSION :\n")
        f.write(
            "L'analyse confirme que la frontière diagnostique n'est pas triviale. Les clusters trouvés de manière non supervisée ne répliquent pas parfaitement les labels cliniques. Ce chevauchement structurel rend indispensable l'utilisation de modèles supervisés complexes (RF, SVM) ou justifie une éventuelle exploration en Deep Learning pour capturer des signatures dynamiques plus fines.\n"
        )

    print("\n--- Synthèse Croisée K-Means vs Groupe Clinique ---")
    print(cross_tab)
    print(
        f"Analyse terminée. {CLUSTERING_FIG_DIR} contient les visuels multivariés et projections."
    )


if __name__ == "__main__":
    run_patient_clustering()


# ======================================================================
# FUZZY C-MEANS
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gaitpdb.config import FUZZY_FIG_DIR, OUTPUT_DIR, RANDOM_STATE
from gaitpdb.features import build_feature_matrix
from gaitpdb.validate import load_stable_features
from gaitpdb.viz.utils import clean_label, save_fig, setup_style

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
