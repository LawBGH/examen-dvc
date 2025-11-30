#!/usr/bin/env python3
# Évaluation finale du modèle entraîné (régression).
# - Charge models/model.pkl (dict contenant 'model' et 'numeric_cols' ou modèle seul)
# - Charge data/processed/X_test_scaled.csv et data/processed/y_test.csv
# - Prédit, calcule MSE et R2
# - Écrit metrics/scores.json et data/processed/predictions.csv
#
# Remarques :
# - Le fichier de sortie des prédictions s'appelle exactement "prediction.csv"
# - Le fichier de métriques s'appelle exactement "scores.json"

import json
import pickle
import yaml
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    # Charge la configuration
    params = read_yaml("params.yaml")
    ds = params["dataset"]
    paths = params["paths"]

    # Chemins attendus (adaptés à params.yaml)
    x_test_scaled_path = Path(ds.get("X_test_scaled", "data/processed/X_test_scaled.csv"))
    y_test_path = Path(ds["y_test"])
    model_path = Path(paths["models_dir"]) / "gbr_model.pkl"
    metrics_dir = Path(paths["metrics_dir"])
    preds_out_path = Path("data/processed/predictions.csv")  # nom demandé par l'utilisateur
    scores_out_path = metrics_dir / "scores.json"          # nom demandé par l'utilisateur

    # Vérifications basiques
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}. Exécute l'entraînement d'abord.")
    if not x_test_scaled_path.exists():
        raise FileNotFoundError(f"X_test_scaled introuvable : {x_test_scaled_path}. Exécute la normalisation d'abord.")
    if not y_test_path.exists():
        raise FileNotFoundError(f"y_test introuvable : {y_test_path}. Exécute la séparation des données d'abord.")

    # Chargement des données
    X_test = pd.read_csv(x_test_scaled_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # Chargement du modèle (et des colonnes numériques utilisées si présentes)
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    # Supporte deux formats possibles :
    # - dict {"model": model, "numeric_cols": [...]}
    # - objet modèle directement
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        numeric_cols = model_obj.get("numeric_cols", None)
    else:
        model = model_obj
        numeric_cols = None

    # Si numeric_cols est présent, on vérifie qu'elles existent dans X_test
    if numeric_cols:
        missing = [c for c in numeric_cols if c not in X_test.columns]
        if missing:
            raise ValueError(f"Colonnes numériques attendues manquantes dans X_test_scaled: {missing}")
        X_test_used = X_test[numeric_cols]
    else:
        # Sinon on utilise toutes les colonnes du X_test chargé
        X_test_used = X_test

    # Prédictions
    y_pred = model.predict(X_test_used)

    # Calcul des métriques
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Écriture des métriques (scores.json)
    ensure_dir(metrics_dir)
    with open(scores_out_path, "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=2)

    # Écriture des prédictions (predictions.csv) : colonnes y_true, y_pred
    ensure_dir(preds_out_path.parent)
    preds_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_pred": pd.Series(y_pred).reset_index(drop=True)})
    preds_df.to_csv(preds_out_path, index=False)

    # Résumé console
    print(f"Évaluation terminée. MSE={mse:.6f}, R2={r2:.6f}")
    print(f"Prédictions écrites dans {preds_out_path}, métriques dans {scores_out_path}")

if __name__ == "__main__":
    main()
