#!/usr/bin/env python3
# Grid search pour trouver les meilleurs hyperparamètres (régression).
# Lit X_train_scaled et y_train, exécute GridSearchCV,
# écrit metrics/grid_search.json, models/best_params.pkl et models/selected_model.pkl,
# et exporte un rapport CSV dans reports/grid_search_report.csv.

import pickle
import json
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def read_yaml(path: str):
    """Lit un fichier YAML et retourne un dictionnaire Python."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    """Crée le répertoire p s'il n'existe pas."""
    p.mkdir(parents=True, exist_ok=True)

def main():
    # Charge la configuration globale
    params = read_yaml("params.yaml")

    # Raccourcis vers les sections utiles
    ds = params["dataset"]
    paths = params["paths"]
    # accepte soit "grid_search" soit "gridsearch" dans params.yaml
    gs_conf = params.get("grid_search", params.get("gridsearch", {}))
    if not gs_conf:
        raise KeyError("Aucune configuration 'grid_search' ou 'gridsearch' trouvée dans params.yaml")

    # Charge les données normalisées d'entraînement et la cible
    X_train = pd.read_csv(ds["X_train_scaled"])
    y_train = pd.read_csv(ds["y_train"]).squeeze("columns")

    # Ne garder que les colonnes numériques (évite les erreurs sur datetimes / strings)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique détectée dans X_train_scaled.")
    X_train_num = X_train[numeric_cols]

    # Instancie le modèle de base (GradientBoostingRegressor compatible avec learning_rate)
    estimator = GradientBoostingRegressor(**params.get("model", {}).get("base_params", {}))

    # Configure GridSearchCV
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=gs_conf["param_grid"],
        cv=gs_conf.get("cv", 5),
        scoring=gs_conf.get("scoring", "neg_mean_squared_error"),
        n_jobs=gs_conf.get("n_jobs", None),
        refit=gs_conf.get("refit", True),
        return_train_score=False,
        error_score="raise",  # lève les erreurs pour faciliter le debug si un fit échoue
    )

    # Exécute la recherche d'hyperparamètres en utilisant uniquement les colonnes numériques
    grid.fit(X_train_num, y_train)

    # Récupère les meilleurs paramètres et score CV
    best_params = grid.best_params_
    cv_best_score = float(grid.best_score_)

    # Sauvegarde des métriques (CV) et des meilleurs paramètres
    metrics_dir = Path(paths["metrics_dir"])
    ensure_dir(metrics_dir)
    with open(metrics_dir / "grid_search.json", "w") as f:
        json.dump({"cv_best_score": cv_best_score, "best_params": best_params}, f, indent=2)

    models_dir = Path(paths["models_dir"])
    ensure_dir(models_dir)
    # best_params.pkl pour réutilisation
    with open(models_dir / "best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    # selected_model.pkl : meilleur estimateur (ré-entraîné si refit=True)
    # Note: le meilleur estimateur a été entraîné sur X_train_num (colonnes numériques).
    # Si tu veux conserver le modèle avec toutes les colonnes, entraîne à nouveau sur X_train complet après sélection.
    with open(models_dir / "selected_model.pkl", "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    # Rapport CSV détaillé des résultats de la CV
    reports_dir = Path(paths.get("reports_dir", "reports"))
    ensure_dir(reports_dir)
    pd.DataFrame(grid.cv_results_).to_csv(reports_dir / "grid_search_report.csv", index=False)

    print("Grid search terminé. Meilleurs paramètres et modèle sauvegardés.")
    print(f"Colonnes numériques utilisées pour l'entraînement: {numeric_cols}")

if __name__ == "__main__":
    main()
