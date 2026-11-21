"""
Climate Risk AI Engine - Probabilistic & Regression Risk Modeling
Author: Olayiwola Oladimeji Austin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class ClimateVulnerabilityModel:
    """Predictive framework for regional climate variability and risk indexing."""

    def __init__(self):
        self.model = LinearRegression()

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

    def compute_risk_index(self, features: pd.DataFrame) -> np.ndarray:
        raw_scores = self.model.predict(features)
        min_val, max_val = raw_scores.min(), raw_scores.max()
        if max_val == min_val:
            return np.zeros_like(raw_scores)
        return (raw_scores - min_val) / (max_val - min_val)
