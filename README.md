
---

## Étapes du pipeline

### 1. **Download**
- Récupère les données brutes et les place dans `data/raw_data/`.

### 2. **Split**
- Sépare les données en **train** et **test**.
- Génère :
  - `X_train.csv`, `y_train.csv`
  - `X_test.csv`, `y_test.csv`

### 3. **Grid Search**
- Teste plusieurs hyperparamètres pour trouver la meilleure configuration du modèle.
- Sauvegarde :
  - `metrics/grid_search.json`
  - `models/best_params.pkl`
  - `reports/grid_search_report.csv`

### 4. **Training**
- Entraîne le modèle final avec les meilleurs hyperparamètres.
- Produit `models/model.pkl`.

### 5. **Evaluate**
- Évalue le modèle sur les données de test.
- Génère :
  - `metrics/scores.json` (MSE, R², etc.)
  - `data/processed/prediction.csv`

---

## Métriques utilisées

- **MSE (Mean Squared Error)** : mesure l’erreur moyenne au carré entre prédictions et valeurs réelles. Plus petit = meilleur.
- **R² (Coefficient de détermination)** : mesure la proportion de variance expliquée par le modèle. Plus proche de 1 = meilleur.

---

## Utilisation

### Reproduire le pipeline
```bash
dvc repro
