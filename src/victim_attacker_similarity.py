"""
Week 5 Extension: computes a similarity score for each victim-attacker
pair -- Euclidean distance between their two enrollment centroids in
the RobustScaler-normalized feature space -- to test directly whether
behavioral similarity predicts undefended attack severity. If it does,
that's mechanistic confirmation that the defense's failure on
high-severity victims isn't (only) a threshold-tuning problem: a
well-matched adversary's data may not be statistically distinguishable
from genuine variation for that specific pair, which no purely
distributional threshold -- global or per-victim -- can fully solve.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler


def compute_pair_similarity(victim_enroll, attacker_enroll):
    """
    Lower distance = more behaviorally similar pair. Fits a scaler on
    the POOLED victim+attacker enrollment data for this pair only, so
    the distance is measured in a consistently-normalized space rather
    than raw seconds, where some features could dominate purely due to
    scale.
    """
    pooled = np.vstack([victim_enroll, attacker_enroll])
    scaler = RobustScaler().fit(pooled)
    victim_scaled = scaler.transform(victim_enroll)
    attacker_scaled = scaler.transform(attacker_enroll)

    victim_centroid = victim_scaled.mean(axis=0)
    attacker_centroid = attacker_scaled.mean(axis=0)
    return float(np.linalg.norm(victim_centroid - attacker_centroid))
