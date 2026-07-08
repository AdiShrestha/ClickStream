"""
Fits a RobustScaler on background subjects enrollment data only,
consistent with Week 2s RobustScaler decision (Week 1 confirmed real
outliers that distort StandardScalers mean and std). Fitting only on
background data and never held-out data avoids leaking held-out
subjects feature-scale statistics into anything the model touches.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler


def fit_sequence_scaler(sequence_data, subjects, background_subject_ids):
    mask = np.isin(subjects, background_subject_ids)
    n_keys, n_feat = sequence_data.shape[1], sequence_data.shape[2]
    flat = sequence_data[mask].reshape(-1, n_keys * n_feat)
    scaler = RobustScaler()
    scaler.fit(flat)
    return scaler


def apply_sequence_scaler(sequence_data, scaler):
    n_samples, n_keys, n_feat = sequence_data.shape
    flat = sequence_data.reshape(n_samples, n_keys * n_feat)
    scaled_flat = scaler.transform(flat)
    return scaled_flat.reshape(n_samples, n_keys, n_feat)
