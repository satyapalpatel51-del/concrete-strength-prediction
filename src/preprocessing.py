"""
preprocessing.py
Functions to load, clean, and prepare the concrete dataset for modeling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/concrete_data.csv"):
    """Load the cleaned concrete dataset from CSV."""
    return pd.read_csv(path)


def split_and_scale(df, target_col="Concrete compressive strength", test_size=0.2, random_state=42):
    """Split into train/test sets and scale features using StandardScaler.
    Returns X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler