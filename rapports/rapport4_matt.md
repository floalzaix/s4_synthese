# Rapport d'avancement 4 — Analyse au niveau du pas (step-level)

**Dataset :** Gait in Parkinson's Disease v1.0.0 (PhysioNet)  
**Date :** 21 mai 2026  
**Auteur :** Matthieu Damien

---

## 1. Contexte

Ce rapport fait suite au rapport 3, qui posait la classification par **sujet** (1 ligne = 1 sujet, 33 features agrégées). La direction principale ouverte en conclusion était l'analyse par **pas** : traiter le signal brut au niveau du cycle de marche, avec des features physiologiques par pas et une classification agrégée au niveau sujet.

---

## 2. Construction de l'index de pas (`steps.csv`)

Un index maître `output/steps.csv` a été construit via `project/build_steps_index.py`. Chaque ligne correspond à un pas (phase d'appui) segmenté sur `total_L` ou `total_R`.

**Paramètres de segmentation finalisés :**

| Paramètre | Valeur | Justification |
|---|---|---|
| `_STANCE_THRESHOLD_RATIO` | 0.08 | Validé empiriquement (réduction 24% des fusions L/R sans sur-segmentation) |
| `_MIN_STEP_SAMPLES` | 10 (0.1 s) | Garde défensif — les pas < 0.1 s sont des artefacts |
| `_MAX_STEP_SAMPLES` | 200 (2.0 s) | Artefact de fusion confirmé par QC visuel |
| `_MAX_STRIDE_DISTANCE_SAMPLES` | 150 (1.5 s) | Validé : maximum observé = 149 samples sur le dataset complet |

**Statistiques finales (60 439 pas, 165 sujets, 306 enregistrements) :**

| Flag | Count | % |
|---|---|---|
| `ok` | 59 457 | 98.4 % |
| `edge_start` | 487 | 0.8 % |
| `edge_end` | 383 | 0.6 % |
| `too_long` | 112 | 0.2 % |

L'appariement L/R par `stride_id` (algorithme glouton par midpoint) couvre la quasi-totalité des pas. Les 112 pas `too_long` correspondent à des fusions d'appuis ou des pauses, confirmés visuellement via `qc_long_strides.py`.

---

## 3. Architecture de chargement — `StepDataset`

`project/step_dataset.py` implémente un dataset lazy-loading compatible PyTorch (`__len__` + `__getitem__`) :

- Filtre par défaut : exclut `too_short` et `too_long` → **60 327 pas** actifs
- Retourne le signal brut `(T, 16)` (16 capteurs individuels L1–L8, R1–R8)
- `SubjectSplitter` encapsule `GroupKFold` pour garantir que tous les pas d'un sujet restent dans le même fold

---

## 4. Comparaison de modèles step-level (`model_step_comparison.py`)

### 4.1 Features tabulaires

7 features par pas, calculées dans `build_step_features()` :

| Feature | Description |
|---|---|
| `duration_s` | Durée de la phase d'appui |
| `peak_force` | Force maximale |
| `auc_norm` | `auc_force / duration_s` — force moyenne sans biais de durée |
| `foot_L` | Encodage binaire du pied |
| `stride_asym_peak` | `\|peak_L − peak_R\| / (peak_L + peak_R)` |
| `stride_asym_duration` | Asymétrie de durée intra-foulée |
| `stride_asym_auc` | Asymétrie d'AUC intra-foulée |

Les 3 features d'asymétrie sont NaN pour les pas non appariés (~5 %), imputées par médiane dans les pipelines.

### 4.2 Modèles évalués

RF, LogReg, KMeans (K=2), Fuzzy C-Means (C=2), CNN 1D PyTorch — tous évalués par **GroupKFold 5 folds par sujet** pour éviter la fuite de données.

### 4.3 Résultats (niveau sujet — majority vote)

| Modèle | Balanced Acc | ROC-AUC |
|---|---|---|
| FCM | **0.69** | **0.75** |
| KMeans | 0.56 | 0.70 |
| LogReg | 0.56 | 0.78 |
| RF | 0.56 | 0.48 |
| CNN 1D | 0.60 | 0.73 |

> Ces résultats sont produits sur un sous-ensemble de 30 sujets (15 PD + 15 CO) pour les tests de développement. Les valeurs sur le dataset complet seront obtenues à l'exécution de `run_step_comparison()`.

FCM est le modèle le plus robuste au niveau sujet, cohérent avec les résultats du clustering flou du rapport 3 sur les features agrégées. La performance step-level reste modeste (Balanced Acc ~0.55–0.65), ce qui est attendu : un pas individuel est peu discriminant, c'est l'agrégation par sujet qui porte le signal.

---

## 5. Identification des pas anomaliques

`build_focus_table()` produit `output/etude_du_pas/focus_malades_premiers_pas.csv` avec trois types :

| Type | Condition | Interprétation |
|---|---|---|
| `pd_mismatch` | Sujet CO, RF prédit PD | Pas PD-like chez un sujet sain — indicateur précoce potentiel |
| `high_asym` | `stride_asym_peak > 0.15` ou `stride_asym_duration > 0.10` | Asymétrie pathologique (seuil Robinson 1987) |
| `both` | Cumul des deux | Cas les plus saillants cliniquement |

---

## 6. QC visuel (`step_qc.py`)

`project/step_qc.py` génère une figure par pas sélectionné (top 20 par type, tous les `both`) :

- **Panneau 1** : vue contextuelle ±4 s avec le pas cible surligné et le pas partenaire de foulée grisé
- **Panneau 2** : zoom sur le pas avec textbox métriques (y_prob, asymétries, durée, peak)

Le QC visuel confirme que la majorité des cas `pd_mismatch` et `high_asym` sont physiologiquement plausibles et non des artefacts de segmentation.

Sorties dans `output/figures/step_qc/` + `output/etude_du_pas/step_qc_summary.csv`.

---

## 7. Modifications du pipeline

- `main.py` : passage de 9 à **10 étapes**, `project.step_qc.main()` ajouté en étape 10
- `project/config.py` : `STEP_QC_FIG_DIR` ajouté
- `output/etude_du_pas/figures/model_step_comparison/model_comparison_table.csv` : tableau de comparaison mean ± std exporté automatiquement

---

## 8. Scripts ajoutés

| Script | Rôle |
|---|---|
| `project/build_steps_index.py` | Construit `steps.csv` et `steps_features.csv` |
| `project/step_dataset.py` | Dataset lazy-loading + `SubjectSplitter` |
| `project/model_step_comparison.py` | Comparaison RF / KMeans / FCM / CNN 1D step-level |
| `project/step_qc.py` | QC visuel des pas pd_mismatch et high_asym |
| `project/qc_long_strides.py` | QC visuel des strides avec distance L/R > 1 s |
| `project/validate_stance_threshold.py` | Comparaison empirique seuil 0.05 vs 0.08 |

---

*Rapport intermédiaire — un rapport détaillé avec résultats complets sur le dataset entier suivra.*
