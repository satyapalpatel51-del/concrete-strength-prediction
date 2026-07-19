"""
train_model.py
Trains Linear Regression, Random Forest, and XGBoost, compares them,
and saves the best model + scaler to disk.

Run from project root with:  python -m src.train_model
"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocessing import load_data, split_and_scale
from src.evaluate import evaluate_model


def main():
    df = load_data("data/concrete_data.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
    }

    best_model, best_name, best_r2 = None, None, -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name}: {metrics}")

        if metrics["R2"] > best_r2:
            best_r2, best_model, best_name = metrics["R2"], model, name

    print(f"\nBest model: {best_name} (R2 = {best_r2:.4f})")

    joblib.dump(best_model, "src/best_model.pkl")
    joblib.dump(scaler, "src/scaler.pkl")
    print("Saved best_model.pkl and scaler.pkl in src/")


if __name__ == "__main__":
    main()