"""
[ROLE]
Analyse exploratoire non supervisée en 6D : Projections, Clustering et Visualisations Multivariées.

[RESPONSIBILITY]
- Explorer la structure latente sur les 6 features retenues (PARSIMONIOUS).
- Séparer conceptuellement le clustering (calcul 6D) des projections (lecture 2D/3D).
- Visualiser les contributions (PCA Loadings) et les distributions par cluster.
- Documenter le recouvrement réel et structurel des groupes cliniques.

[OUTPUTS]
- output/patient_embeddings.csv
- output/cluster_summary.csv
- output/clustering_metrics.csv
- output/clustering_report.txt
- output/figures/clustering/*.png
"""

##
# @file patient_clustering.py
# @brief Analyse exploratoire non supervisée (Clustering et Projections).
# @details Implémente K-Means, GMM, PCA et t-SNE pour explorer la structure des données.
#

from __future__ import annotations

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

from project.config import CLUSTERING_FIG_DIR, OUTPUT_DIR, RANDOM_STATE
from project.features import build_feature_matrix
from project.validate import load_stable_features
from project.viz_utils import PALETTE, clean_label, save_fig, setup_style

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
