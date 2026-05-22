# GaitPDB — Analyse de la Marche Parkinsonienne

Projet de synthèse ESEO E4 — Classification PD vs CO à partir du dataset  
**Gait in Parkinson's Disease v1.0.0** (PhysioNet).

---

## Objectif

Classifier des sujets **Parkinsoniens (PD)** vs témoins sains **(CO)** à partir de signaux de pression plantaire (8 capteurs par pied, 100 Hz). Le pipeline produit des features biomécaniques agrégées par sujet, les valide par cross-validation robuste, les interprète via XAI/SHAP, et explore la structure latente par clustering.

---

## Arborescence

```
.
├── main.py                    # Orchestrateur unique du pipeline
├── project/
│   ├── config.py              # Constantes, chemins, RANDOM_STATE
│   ├── load.py                # Chargement dataset PhysioNet (.xls, .txt)
│   ├── features.py            # Extraction de features (force, asymétrie, pas)
│   ├── validate.py            # Validation K-Fold + LOSO, sélection intra-CV
│   ├── model_comparison.py    # Benchmark multi-algorithmes (RF, LogReg, SVC, XGBoost...)
│   ├── patient_clustering.py  # Clustering K-Means/GMM + projections PCA/t-SNE
│   ├── fuzzy_clustering.py    # Clustering flou (Fuzzy C-Means)
│   ├── feature_reduction.py   # Comparaison sets de features (complet / réduit / compact)
│   ├── xai.py                 # Importance MDI + permutation, familles physiologiques
│   ├── shap_analysis.py       # Valeurs SHAP multi-fold (TreeExplainer)
│   ├── visual_check.py        # QC visuel segmentation (atlas 4 sujets)
│   ├── viz_advanced.py        # Profils de force, heatmaps asymétrie, distributions
│   ├── viz_bridge.py          # Schémas conceptuels (pont dynamique→statique, pipeline)
│   └── viz_utils.py           # Palette, style, save_fig centralisés
├── datasets/
│   └── gait-in-parkinsons-disease-1.0.0/   # Dataset PhysioNet (non versionné)
├── output/
│   ├── figures/               # Toutes les figures générées (sous-dossiers par module)
│   │   ├── validation/
│   │   ├── model_comparison/
│   │   ├── clustering/
│   │   ├── fuzzy_clustering/
│   │   ├── xai/
│   │   ├── shap/
│   │   ├── segmentation/
│   │   ├── gait_profiles/
│   │   ├── asymmetry_heatmaps/
│   │   └── concept_map/
│   └── *.csv / *.txt          # Métriques, rapports, embeddings exportés
└── rapports/
    ├── rapport2_matt.md
    ├── rapport3_matt.md
    └── images/
```

---

## Description des modules

### `config.py`
Centralise toutes les constantes partagées : `RANDOM_STATE = 42`, `SESSION = "01"` (marche normale), chemins `OUTPUT_DIR` et `FIG_DIR` et leurs sous-dossiers nommés. Tout autre module importe depuis ici, jamais en dur.

### `load.py`
Lit `demographics.xls` et les fichiers signal `.txt` (19 colonnes, 100 Hz). Expose `load_dataset_index()` qui fusionne démographie et chemins de fichiers en un seul DataFrame indexé.

### `features.py`
Cœur de l'extraction. Pour chaque sujet :
- `_force_features()` : statistiques de force, AUC normalisée par durée, asymétrie L/R.
- `_temporal_features()` : détection de pics, cadence, intervalles inter-pas.
- `extract_step_features()` : segmentation stance/swing, asymétries de phase.
- `build_feature_matrix()` : construit le DataFrame final (1 ligne = 1 sujet, session 01).

Exporte les listes `FEATURE_COLS` (33 features), `META_COLS`, `STEP_FEATURES`, `ASYMMETRY_FEATURES`.

### `validate.py`
**Module central de validation.** `run_validation(df)` :
1. K-Fold stratifié (5 folds) avec `Pipeline(SelectFromModel(RF100) → RF200)` — la sélection de features est apprise sur le fold d'entraînement uniquement (pas de fuite).
2. LOSO (Leave-One-Study-Out) : robustesse inter-cohortes.
3. Diagnostics visuels : courbe ROC, Precision-Recall, matrice de confusion.
4. Exporte `cv_feature_stability.csv` (fréquence de sélection par feature).

`load_stable_features(min_freq=0.6)` : interface publique pour récupérer les features stables depuis le CSV produit.

### `model_comparison.py`
Benchmark de 5+ algorithmes (RF, GradBoost, LogReg, LinearSVC, SVC-RBF, XGBoost optionnel) sur `FINAL_FEATURES` en 5-Fold CV. Génère barplots et boxplots par métrique (Accuracy, F1, ROC-AUC, Balanced Accuracy).

### `patient_clustering.py`
Analyse exploratoire non supervisée : K-Means (k=2) et GMM sur les features stables (issues de `load_stable_features()`). Projections PCA 2D/3D, t-SNE, UMAP (optionnel). Calcul des métriques Silhouette, Davies-Bouldin, Calinski-Harabasz. Export des embeddings dans `patient_embeddings.csv`.

### `fuzzy_clustering.py`
Fuzzy C-Means dans l'espace des features stables. Sélection automatique de `C_opt` (2 à 6) par maximisation de la Fuzzy Partition Coefficient. Export des degrés d'appartenance (`u_final.csv`) et rapport interprétif. **Outil exploratoire uniquement — ne produit pas de métriques de classification.**

### `feature_reduction.py`
Compare trois jeux de features (complet ~19, réduit bilatéral ~13, compact clinique 4) en 5-Fold CV RF. Évalue l'impact de la simplification sur la performance et la stabilité XAI (coefficient de variation inter-folds des importances par permutation).

### `xai.py`
Importance MDI et permutation moyennées sur 5 folds, catégorisées par famille physiologique (Asymétrie, Variabilité, Phases, Cadence). Intègre optionnellement `shap_analysis.py` pour un troisième estimateur. Génère `feature_importance_metrics.csv` et `xai_report.txt`.

### `shap_analysis.py`
Analyse SHAP multi-fold via `TreeExplainer`. Les valeurs SHAP sont calculées sur le fold de test uniquement. Exporte `shap_values.csv`, `shap_summary.csv`, et 3 figures (barplot, stabilité inter-folds, heatmap groupe).

### `visual_check.py`
QC visuel de la segmentation des pas sur 4 sujets représentatifs. Génère un atlas 3-vues (vue longue, zoom précision, overlay L/R) par sujet.

### `viz_advanced.py`
Profils de force consolidés (normalisés à 100 échantillons) et heatmaps d'asymétrie dynamique PD vs CO. Boxplots stratifiés par étude pour les features clés.

### `viz_bridge.py`
Schémas conceptuels matplotlib : pont dynamique→statique (préfigure la phase COP) et architecture logique du pipeline.

---

## Lancer le pipeline

```bash
# Depuis la racine du projet
python main.py
```

Le pipeline s'exécute en 9 étapes dans l'ordre suivant :
1. Chargement unique de la matrice de features (session 01)
2. Validation K-Fold + LOSO
3. Benchmark multi-algorithmes
4. Clustering K-Means/GMM + projections
5. Fuzzy C-Means
6. Réduction de l'espace features (Phase 4)
7. XAI (importance + SHAP si installé)
8. QC segmentation visuelle
9. Visualisations avancées + schémas conceptuels

Toutes les sorties (figures `.png`, métriques `.csv`, rapports `.txt`) sont générées dans `output/`.

### Dépendances principales
```
numpy >= 2.0     # np.trapezoid requis (sinon numpy >= 1.20 avec np.trapz)
pandas
scipy
scikit-learn
matplotlib
seaborn
skfuzzy          # pip install scikit-fuzzy
shap             # optionnel — pip install shap
xgboost          # optionnel
umap-learn       # optionnel — pip install umap-learn
xlrd             # pour lire demographics.xls
```

---

## État du projet

| Module | État |
|---|---|
| Chargement & features | Stable |
| Validation K-Fold / LOSO | Stable |
| Benchmark modèles | Stable |
| XAI (MDI + permutation) | Stable |
| SHAP multi-fold | Stable (dépend de `shap`) |
| Clustering patient (K-Means, GMM) | Stable |
| Fuzzy clustering | Stable |
| Réduction de features | Stable |
| QC segmentation | Stable |
| Visualisations avancées | Stable |

### Prochaines étapes (Phase suivante)
- Extraction et classification au niveau du **pas individuel** (step-level features)
- Clustering de pas (séparation marche normale vs altérée intra-sujet)
- Architecture réseau de neurones pour séquences de pas (LSTM, 1D-CNN)
- Analyse du Centre de Pression (COP) — déjà esquissée dans `viz_bridge.py`

---

## Dataset

**Gait in Parkinson's Disease — PhysioNet v1.0.0**  
Cohortes : Ga (Galveston), Ju (Jülich), Si (Siegburg) — 3 études indépendantes  
Groupes : 93 sujets PD, 73 témoins CO  
Signal : 16 capteurs de pression (8L + 8R) + total L/R + temps — 100 Hz
