"""
Confirms train_encoder actually raises on non-finite loss rather than
silently continuing. This deliberately forces the failure condition.
It is testing the safety net itself, not trying to train a good model.
"""
import numpy as np
import pytest
from src.train_encoder import train_encoder


def test_train_encoder_raises_on_nan_input():
    rng = np.random.RandomState(0)
    n_keys = 4
    subjects = np.array([f"s{i}" for i in range(3) for _ in range(10)])
    sequence_data = rng.normal(size=(30, n_keys, 3)).astype(np.float32)
    # Inject NaN into every sample, which propagates through LSTM and loss
    sequence_data[:] = np.nan
    background_ids = ["s0", "s1", "s2"]

    with pytest.raises(RuntimeError, match="non-finite"):
        train_encoder(
            sequence_data, subjects, background_ids,
            device="cpu", epochs=2, steps_per_epoch=2,
            batch_size=8, lr=1e-3, seed=0,
        )
