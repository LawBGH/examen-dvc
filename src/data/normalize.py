#!/usr/bin/env python3
"""
Normalisation robuste :
- détecte et convertit les colonnes datetime en timestamp (float seconds)
- normalise les colonnes numériques (après conversion)
- écrit X_train_scaled.csv et X_test_scaled.csv
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def convert_datetime_cols(df: pd.DataFrame):
    # détecte colonnes parseables en datetime et les convertit en timestamp (float seconds)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                ser = pd.to_datetime(df[col], errors="raise")
            except Exception:
                continue
            # conversion réussie -> remplacer par timestamp en secondes (float)
            df[col] = ser.view("int64") / 1e9
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")

    # Convertit les colonnes datetime si présentes
    X_train = convert_datetime_cols(X_train)
    X_test = convert_datetime_cols(X_test)

    # Détecte colonnes numériques après conversion
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique détectée après conversion.")

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_num = scaler.transform(X_test[numeric_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = X_train_num
    X_test_scaled[numeric_cols] = X_test_num

    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test_scaled.csv", index=False)

    print("Normalisation terminée (datetimes convertis en timestamp si présents).")

if __name__ == "__main__":
    main()
