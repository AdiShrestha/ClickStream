import numpy as np
from src.cusum_defense import DefendedAdaptiveBaseline


def test_cusum_accumulates_for_persistent_directional_drift():
    rng = np.random.RandomState(0)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = DefendedAdaptiveBaseline(cusum_k=0.0, cusum_h=1e9).initialize(enrollment)

    for i in range(20):
        drifting_candidate = rng.normal(0, 1, size=5) + (i * 0.3)
        baseline.offer_candidate(drifting_candidate, round_index=i)

    assert baseline.cusum_history[-1] > baseline.cusum_history[0], (
        "CUSUM should accumulate upward under persistent directional drift"
    )


def test_defense_blocks_absorption_when_triggered():
    rng = np.random.RandomState(1)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = DefendedAdaptiveBaseline(cusum_k=0.0, cusum_h=0.5).initialize(enrollment)

    for i in range(10):
        drifting_candidate = rng.normal(0, 1, size=5) + (i * 2.0)  # aggressive, fast drift
        baseline.offer_candidate(drifting_candidate, round_index=i)

    assert len(baseline.defense_triggered_rounds) > 0, (
        "With aggressive drift and a low h, the defense should trigger at least once"
    )
    for triggered_round in baseline.defense_triggered_rounds:
        assert not baseline.history[triggered_round].absorbed, (
            f"Round {triggered_round} was flagged but still shows as absorbed -- "
            "the gate is not actually blocking absorption when triggered"
        )


def test_stable_candidates_do_not_trigger_defense():
    rng = np.random.RandomState(2)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = DefendedAdaptiveBaseline(cusum_k=0.5, cusum_h=5.0).initialize(enrollment)

    for i in range(50):
        stable_candidate = rng.normal(0, 1, size=5)  # same distribution, no drift
        baseline.offer_candidate(stable_candidate, round_index=i)

    assert len(baseline.defense_triggered_rounds) == 0, (
        "Non-drifting candidates from the enrolled distribution should not "
        "trigger the defense with a reasonable k/h"
    )
