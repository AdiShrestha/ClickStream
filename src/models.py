"""
Week 2: per-user anomaly-detection model wrapper. One instance is
fit per enrolled subject, on that subject's own enrollment data only.
This is the verification (1:1) framing, never population-level.

RobustScaler is used deliberately over StandardScaler: Week 1 confirmed
real outliers up to 2.0353s (hold) and 25.9873s (DD) in this dataset.
StandardScaler's mean and std are themselves distorted by such outliers;
RobustScaler's median/IQR-based scaling is far less affected.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


class PerUserModel:
    """
    algorithm: "isolation_forest" or "one_class_svm"
    contamination: expected fraction of anomalies (used as nu for OC-SVM)
    random_state: for reproducibility (IsolationForest only)
    """

    def __init__(self, algorithm="isolation_forest", contamination=0.05, random_state=42):
        self.algorithm = algorithm
        self.scaler = RobustScaler()
        if algorithm == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=200, contamination=contamination, random_state=random_state
            )
        elif algorithm == "one_class_svm":
            self.model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, X_enroll: np.ndarray) -> "PerUserModel":
        X_scaled = self.scaler.fit_transform(X_enroll)
        self.model.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more consistent with the enrolled baseline."""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
