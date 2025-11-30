import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--target_col", type=str, default="silica_concentrate")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Charger les paramètres
    with open(args.params) as f:
        params = yaml.safe_load(f)
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # Lire les données
    df = pd.read_csv(args.input)
    if args.target_col not in df.columns:
        raise ValueError(f"Target '{args.target_col}' not found in columns: {list(df.columns)}")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(args.output, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.output, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(args.output, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(args.output, "y_test.csv"), index=False)

    print(f"[split] X_train={X_train.shape}, X_test={X_test.shape}")

if __name__ == "__main__":
    main()
