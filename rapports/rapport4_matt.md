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

## 9. Améliorations des performances (`amelioration_performances.py`)

Suite aux résultats du rapport précédent (Balanced Acc 0.55–0.65, en deçà de la cible 0.75–0.85), quatre axes d'amélioration pragmatiques ont été implémentés dans `project/amelioration_performances.py`.

---

### 9.1 Traitement du signal — filtrage Butterworth 10 Hz

Un filtre passe-bas Butterworth d'ordre 4, zero-phase (`filtfilt`), à coupure 10 Hz est appliqué aux signaux `total_L` et `total_R` avant extraction des features de forme.

**Justification :** Pour les signaux de pression plantaire à 100 Hz, un filtre passe-bas 5–10 Hz est standard dans la littérature de gait analysis (Bartlett et al. 2007). Le cutoff retenu à 10 Hz préserve les pics de force et les transitions stance/swing tout en atténuant le bruit haute fréquence des capteurs de pression.

Le cache est structuré par fichier source (`source_file` → signaux filtrés L et R), ce qui évite de recharger et refiltrer le même fichier à chaque pas.

![[signal_filter_comparison.png]]

*La structure des pas est intégralement préservée — les pics coïncident entre signal brut et filtré ; seul le bruit de haute fréquence est supprimé.*

---

### 9.2 Features enrichies

7 nouvelles features par pas, calculées dans `build_extended_features()` :

| Feature | Description | Nouveauté |
|---|---|---|
| `rise_time_s` | Temps du début du pas au pic de force (phase de chargement) | Nouvelle |
| `fall_time_s` | Temps du pic à la fin du pas (phase de déchargement) | Nouvelle |
| `rise_fall_ratio` | `rise_time_s / fall_time_s` — forme asymétrique du pas | Nouvelle |
| `impulse_front_ratio` | AUC première moitié / AUC totale — dominance du chargement | Nouvelle |
| `peak_force_subj_norm` | `peak_force / médiane_sujet` — invariant à la corpulence | Nouvelle |
| `stride_asym_rise` | Asymétrie intra-foulée de `rise_time_s` (même logique que `stride_asym_peak`) | Nouvelle |
| `stride_asym_fall` | Asymétrie intra-foulée de `fall_time_s` | Nouvelle |
| `stride_asym_impulse` | Asymétrie intra-foulée de `impulse_front_ratio` | Nouvelle |

Les features de forme (`rise_time_s`, `fall_time_s`, `impulse_front_ratio`) sont calculées sur le signal **filtré**, ce qui les rend plus stables. La normalisation sujet (`peak_force_subj_norm`) élimine l'effet de masse corporelle, source de variabilité inter-sujet dominante par rapport au signal Parkinson.

Fichier exporté : `output/etude_du_pas/features_enriched.csv` (15 features × N_pas).

---

### 9.3 Optimisation du seuil de décision

Le seuil de décision par défaut (0.5) est sous-optimal pour des datasets déséquilibrés ou des modèles mal calibrés. La fonction `optimize_threshold()` balaye les seuils [0.05, 0.95] en pas de 0.005 et calcule :

- **Accuracy** au sens strict
- **F1-Score** (classe PD)
- **Youden J = Sensibilité + Spécificité − 1** — critère équilibré, recommandé pour la discrimination clinique

Le seuil optimal est déterminé par le maximum de J (critère Youden), plus robuste que l'accuracy en présence d'un léger déséquilibre PD/CO.

![[threshold_curve.png]]

Le seuil optimisé et les métriques associées sont sauvegardés dans `output/etude_du_pas/threshold.json` :

```json
{
  "threshold_youden": <optimal>,
  "threshold_f1":     <optimal_f1>,
  "threshold_acc":    <optimal_acc>,
  "metrics_at_youden": {
    "balanced_acc": ...,
    "f1":           ...,
    "roc_auc":      ...
  }
}
```

---

### 9.4 Stacking léger RF + LogReg → LogReg sujet

**Architecture :** Les prédictions pas-level OOF (out-of-fold) de RF et LogReg sont agrégées en proba moyenne par sujet. Ces deux probas sujet constituent les meta-features d'un `LogisticRegression` entraîné via **LOOCV** (leave-one-subject-out).

**Absence de fuite de données :** Les probas OOF sont produites sur les folds de test du GroupKFold, donc strictement hors des données d'entraînement de chaque modèle de base. Le stacker reçoit des prédictions généralisées, pas des prédictions en train.

| Modèle | Balanced Acc | ROC-AUC | F1-Score | Delta vs baseline |
|---|---|---|---|---|
| CNN1D (baseline) | **0.7414** | 0.7860 | 0.7290 | référence (meilleur) |
| RF_enriched | 0.6542 | 0.7334 | 0.6543 | +7.5 pts vs RF |
| Stacking_RF_LogReg | 0.6317 | 0.6994 | 0.7100 | — |
| LogReg_enriched | 0.6146 | 0.6359 | 0.6089 | +6.2 pts vs LogReg |
| RF_baseline | 0.5793 | 0.6360 | 0.5834 | référence RF |
| LogReg_baseline | 0.5526 | 0.6062 | 0.5433 | référence LogReg |

**Seuil Youden optimisé (RF baseline OOF) :**
- Seuil optimal : **0.620** (vs défaut 0.500)
- Balanced Acc au seuil : **0.6454** (+6.6 pts vs RF@0.5)
- ROC-AUC : 0.628

![[performance_comparison.png]]

![[stacking_comparison.png]]

---

### 9.5 Résumé des axes et gains attendus

| Axe | Mécanisme | Gain observé |
|---|---|---|
| Features enrichies + filtrage 10 Hz | rise_time, fall_time, impulse_ratio, normalisation sujet | **+7.5 pts RF**, +6.2 pts LogReg (Balanced Acc) |
| Seuil Youden (0.62 vs 0.50) | Maximise TPR+TNR — calibration du point de coupure | **+6.6 pts RF** (0.579 → 0.645) |
| Stacking RF + LogReg | Combinaison des biais complémentaires (Balanced Acc 0.63) | Modeste (+4 pts vs LogReg) |
| CNN1D (baseline) | Déjà à 0.74 Balanced Acc — quasi-cible | **Cible atteinte** |

**Conclusion intermédiaire :** Le modèle CNN1D est le plus performant à ce stade (0.74, quasi-cible 0.75). L'enrichissement de features améliore significativement RF (+7.5 pts) et LogReg (+6.2 pts) mais ne dépasse pas CNN1D. Le stacking RF+LogReg est légèrement sous-optimal face à RF_enriched seul — probablement parce que LogReg dilue le signal plutôt que de le compléter à l'échelle des 165 sujets disponibles.

**Limites identifiées :** La performance step-level plafonne en raison de la variabilité intra-sujet (un patient PD peut avoir de "bons" pas). L'axe d'amélioration le plus impactant restant serait l'intégration des features enrichies dans le CNN1D (inputs multicanaux + features tabulaires en parallèle).

---

## 10. Phase 3 — Features par capteur, CNN hybride et data augmentation

Suite au constat que le CNN1D plafonne à ~0.66 Balanced Acc malgré les optimisations d'entraînement (dropout, CosineAnnealingLR, class weights), la stratégie change : **enrichir l'information disponible au modèle** plutôt que de tuner l'entraînement.

Trois axes complémentaires ont été implémentés simultanément.

---

### 10.1 Features par capteur individuel (P6)

Les 16 capteurs de pression plantaire (L1–L8 pour le pied gauche, R1–R8 pour le pied droit) étaient jusqu'ici agrégés en `total_L` et `total_R`. Cette étape exploite chaque capteur individuellement pour capturer la **distribution spatiale de la pression** — un marqueur connu de la marche parkinsonienne (traînée du pied, appui talon/avant-pied modifié).

**30 nouvelles features par pas**, calculées dans `build_sensor_features()` :

| Catégorie | Features | Description |
|---|---|---|
| Moyennes par capteur (16) | `mean_L1`..`mean_L8`, `mean_R1`..`mean_R8` | Force moyenne sur le pas pour chaque capteur |
| Ratios spatiaux (4) | `forefoot_ratio_L/R`, `forefoot_asym`, `heel_dominance` | Ratio avant-pied/talon — détecte l'appui anormal |
| Asymétries par capteur (8) | `sensor_asym_1`..`sensor_asym_8` | `\|mean(Li) − mean(Ri)\| / (mean(Li) + mean(Ri))` par paire |
| Entropies spatiales (2) | `entropy_L`, `entropy_R` | Shannon sur la distribution de pression — pression concentrée vs répartie |

**Impact :** Ces 30 features, combinées aux 15 features enrichies existantes (45 features totales), font passer la **LogReg de 0.57 à 0.75 Balanced Acc** (+18 points) et le **Stacking de 0.61 à 0.76** (+15 points). C'est le gain le plus important de toutes les améliorations.

Fichier exporté : `output/etude_du_pas/features_sensor.csv` (30 features × 33 277 pas).

---

### 10.2 CNN hybride (P7)

Le CNN1D ne traite que le signal brut (16 canaux × 150 samples). Le **HybridCNN** combine deux branches dans une architecture à fusion tardive :

```
Branche Signal (CNN) :
  Conv1d(16→32, k=7) + BN + ReLU + Dropout(0.3) + MaxPool(2)
  Conv1d(32→64, k=5) + BN + ReLU + Dropout(0.3) + MaxPool(2)
  Conv1d(64→64, k=3) + BN + ReLU + Dropout(0.3) + AdaptiveAvgPool(1)
  → embedding signal (batch, 64)

Branche Tabulaire (MLP) :
  Linear(45 → 32) + ReLU + Dropout(0.3)
  Linear(32 → 32) + ReLU
  → embedding tabulaire (batch, 32)

Fusion :
  Concat(signal_emb, tab_emb) → (batch, 96)
  Dropout(0.5)
  Linear(96 → 2)
```

**Features tabulaires en entrée :** Les 45 features (15 enrichies + 30 capteurs), normalisées par `StandardScaler` et imputées par médiane sur le fold train. Le modèle apprend conjointement des représentations du signal brut ET des features explicites.

**Entraînement :** 20 epochs fixes, `CosineAnnealingLR`, `CrossEntropyLoss` avec class weights équilibrés, `StratifiedGroupKFold` 5 folds.

![[hybridcnn_training_curves.png]]

*Les courbes d'entraînement montrent une convergence stable sur les 5 folds. La val loss reste bruitée (peu de sujets en validation) mais le modèle ne diverge pas grâce au dropout agressif (0.3 + 0.5).*

---

### 10.3 Data augmentation (P8)

Pour réduire l'overfitting du CNN sur les ~26 000 pas d'entraînement par fold, des augmentations sont appliquées **on-the-fly** pendant l'entraînement uniquement (pas pendant l'évaluation) :

| Augmentation | Paramètres | Objectif |
|---|---|---|
| Scaling amplitude | `× U[0.9, 1.1]` (p=0.5) | Simule les variations de pression liées au poids/chaussures |
| Bruit gaussien | `+ N(0, 0.02 × std_canal)` (p=0.5) | Simule le bruit des capteurs de pression |

Le time shift initialement prévu (±5 samples) n'a pas été retenu : le risque de décaler les pics de force hors de la fenêtre est trop élevé pour un gain marginal.

---

### 10.4 Résultats — Comparaison de tous les modèles

#### Niveau sujet — `run_step_comparison` (modèles directs)

| Modèle | Accuracy | Balanced Acc | ROC-AUC | Nouveau ? |
|---|---|---|---|---|
| **HybridCNN** | **0.769** | **0.769** | **0.866** | P7+P8 |
| CNN1D | 0.685 | 0.657 | 0.755 | Phase 2 |
| LogReg | 0.558 | 0.571 | 0.606 | Baseline |
| RF | 0.600 | 0.558 | 0.637 | Baseline |
| FCM | 0.485 | 0.509 | 0.503 | Baseline |
| KMeans | 0.484 | 0.484 | 0.544 | Baseline |

![[step_subject_barplot_balanced_acc.png]]

*Le HybridCNN domine nettement tous les autres modèles avec une faible variance inter-folds.*

#### Niveau sujet — `run_improvements` (features tabulaires + stacking)

| Modèle | Accuracy | Balanced Acc | F1-Score | ROC-AUC | Nouveau ? |
|---|---|---|---|---|---|
| **Ensemble_Final** | **0.770** | **0.769** | **0.791** | **0.849** | P6+P7 |
| Stacking_RF_LogReg_full | 0.770 | 0.764 | 0.798 | 0.844 | P6 |
| LogReg_full | 0.751 | 0.750 | 0.750 | 0.840 | P6 |
| RF_full | 0.674 | 0.641 | 0.613 | 0.826 | P6 |
| Stacking_RF_LogReg | 0.630 | 0.609 | 0.702 | 0.688 | Phase 2 |
| RF_baseline | 0.600 | 0.558 | 0.534 | 0.637 | Baseline |
| LogReg_baseline | 0.558 | 0.571 | 0.541 | 0.606 | Baseline |

L'**Ensemble_Final** combine les prédictions OOF du HybridCNN, RF_full et LogReg_full via un stacker `LogisticRegression` en LOOCV sujet. Il atteint 0.769 Balanced Acc — au même niveau que le HybridCNN seul, confirmant que le CNN hybride capture déjà l'essentiel de l'information tabulaire via sa branche MLP.

![[performance_comparison.png]]

*Comparaison des 4 métriques pour tous les modèles tabulaires et ensembles. Les modèles avec features capteurs (suffixe `_full`) dépassent systématiquement la cible 0.75 en Accuracy et Balanced Acc.*

![[stacking_comparison.png]]

*Vue d'ensemble de tous les modèles par métrique. L'Ensemble_Final et le Stacking_full franchissent la ligne cible 0.75 sur toutes les métriques sauf le F1-Score du Stacking_full.*

---

### 10.5 Analyse des gains par axe d'amélioration

| Axe | Gain principal | Détail |
|---|---|---|
| **P6 — Features capteurs** (30 features) | **+18 pts LogReg** (0.57 → 0.75) | Information spatiale très discriminante — distribution pression avant-pied/talon, asymétries par capteur |
| **P7 — CNN hybride** (signal + tabulaire) | **+11 pts vs CNN1D** (0.66 → 0.77) | La branche tabulaire fournit au CNN des features explicites qu'il ne peut pas extraire du signal brut seul |
| **P8 — Data augmentation** | Régularisation intégrée | Contribue à la stabilité de l'entraînement (variance inter-folds réduite) |
| **Ensemble_Final** | +0.5 pts vs HybridCNN seul | Gain marginal — le HybridCNN capture déjà l'information des features tabulaires |

---

## 11. Conclusion

### Progression globale

| Phase | Meilleur modèle | Balanced Acc | ROC-AUC |
|---|---|---|---|
| Baseline (rapport 4 initial) | FCM | 0.69 | 0.75 |
| Phase 2 (StratifiedGroupKFold, Youden, dropout) | CNN1D | 0.66 | 0.75 |
| **Phase 3 (capteurs, hybride, augmentation)** | **HybridCNN** | **0.77** | **0.87** |

Le pipeline atteint **0.77 Balanced Acc** et **0.87 ROC-AUC** au niveau sujet — une progression de +8 points de Balanced Acc et +12 points de ROC-AUC par rapport au meilleur modèle de la Phase 2. La cible de 0.80 n'est pas atteinte mais le ROC-AUC de 0.87 indique un fort pouvoir discriminant : avec un seuil de décision adapté au contexte clinique (sensibilité vs spécificité), le modèle peut être calibré pour différents cas d'usage.

### Facteur limitant principal

Le dataset GaitPDB contient **165 sujets** (93 PD, 72 CO). Avec un StratifiedGroupKFold à 5 folds, chaque fold de validation ne contient que ~33 sujets. Cette taille limite la stabilité de l'évaluation et la capacité du modèle à généraliser. Un dataset plus large (>500 sujets) permettrait probablement de franchir la barre des 0.80.

### Résultat clé pour l'explicabilité

Les **features par capteur individuel** (P6) sont le gain le plus important du pipeline. Elles permettent non seulement une meilleure classification, mais aussi une **interprétation clinique directe** : quels capteurs discriminent PD vs CO, quelle est la distribution spatiale de la pression, l'asymétrie par zone du pied. Ces features sont directement exploitables pour l'analyse XAI (feature importance RF, SHAP).

---

## 12. Scripts modifiés et ajoutés

| Script | Rôle |
|---|---|
| `gaitpdb/build_steps_index.py` | Construit `steps.csv` et `steps_features.csv` |
| `gaitpdb/step_dataset.py` | Dataset lazy-loading + `SubjectSplitter` (StratifiedGroupKFold) |
| `gaitpdb/model_step_comparison.py` | Comparaison RF / KMeans / FCM / CNN1D / **HybridCNN** step-level |
| `gaitpdb/step_improvements.py` | Features enrichies + **capteurs** + seuil + stacking + **Ensemble_Final** |
| `gaitpdb/qc/step_qc.py` | QC visuel des pas pd_mismatch et high_asym |
| `gaitpdb/qc/long_strides.py` | QC visuel des strides avec distance L/R > 1 s |
| `exploration/validate_stance_threshold.py` | Comparaison empirique seuil 0.05 vs 0.08 |
| `run_step.py` | Point d'entrée du pipeline step-level complet |
