#!/usr/bin/env python3
# Entraînement final du modèle de régression.
# - Lit models/best_params.pkl (résultat du grid search)
# - Lit data/processed/X_train_scaled.csv et data/processed/y_train.csv
# - Sélectionne automatiquement les colonnes numériques (cohérent avec grid_search)
# - Entraîne GradientBoostingRegressor avec les meilleurs hyperparamètres
# - Sauvegarde le modèle entraîné dans models/gbr_model.pkl

import pickle
import yaml
import pandas as pd
from pathlib import Path
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
    # Charge la configuration
    params = read_yaml("params.yaml")

    # Raccourcis vers les chemins
    ds = params["dataset"]
    paths = params["paths"]

    # Chemins d'entrée/sortie
    x_train_scaled_path = Path(ds["X_train_scaled"])
    y_train_path = Path(ds["y_train"])
    best_params_path = Path(paths["models_dir"]) / "best_params.pkl"
    model_out_path = Path(paths["models_dir"]) / "gbr_model.pkl"

    # Vérifications basiques
    if not x_train_scaled_path.exists():
        raise FileNotFoundError(f"{x_train_scaled_path} introuvable. Exécute la normalisation d'abord.")
    if not y_train_path.exists():
        raise FileNotFoundError(f"{y_train_path} introuvable. Exécute la séparation des données d'abord.")
    if not best_params_path.exists():
        raise FileNotFoundError(f"{best_params_path} introuvable. Exécute le grid search d'abord.")

    # Chargement des données
    X_train = pd.read_csv(x_train_scaled_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")

    # Sélectionne les colonnes numériques (cohérence avec grid_search)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique détectée dans X_train_scaled. Vérifie la normalisation.")
    X_train_num = X_train[numeric_cols]

    # Charge les meilleurs hyperparamètres
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    # Construit et entraîne le modèle final
    base_params = params.get("model", {}).get("base_params", {})
    # best_params peut contenir des types non sérialisables pour l'init; on les passe tels quels
    model = GradientBoostingRegressor(**best_params, **base_params)
    model.fit(X_train_num, y_train)

    # Sauvegarde du modèle entraîné
    ensure_dir(model_out_path.parent)
    with open(model_out_path, "wb") as f:
        pickle.dump({"model": model, "numeric_cols": numeric_cols}, f)

    print(f"Entraînement terminé. Modèle sauvegardé dans {model_out_path}")
    print(f"Colonnes numériques utilisées: {numeric_cols}")

if __name__ == "__main__":
    main()
