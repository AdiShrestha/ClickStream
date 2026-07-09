"""
Verifies the adaptation mechanism itself, independent of any attack --
this must be correct before an attack result built on top of it means
anything.
"""
import numpy as np
from src.adaptive_baseline import AdaptiveBaseline


def test_initialize_fits_on_given_enrollment():
    rng = np.random.RandomState(0)
    enrollment = rng.normal(0, 1, size=(50, 5))
    baseline = AdaptiveBaseline().initialize(enrollment)
    assert baseline.enrollment.shape == (50, 5)


def test_offer_candidate_similar_to_enrollment_is_absorbed():
    rng = np.random.RandomState(1)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)

    similar_candidate = rng.normal(0, 1, size=(5,))
    result = baseline.offer_candidate(similar_candidate, round_index=0)
    assert result.absorbed, (
        "A same-distribution candidate should typically clear the "
        "absorption threshold -- if this fails often, the threshold "
        "percentile may be miscalibrated."
    )
    assert baseline.enrollment.shape[0] == 101


def test_offer_candidate_wildly_different_is_rejected():
    rng = np.random.RandomState(2)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)

    extreme_candidate = np.full(5, 100.0)
    result = baseline.offer_candidate(extreme_candidate, round_index=0)
    assert not result.absorbed, "An extreme outlier must not be absorbed"
    assert baseline.enrollment.shape[0] == 100, (
        "Enrollment set must be UNCHANGED when a candidate is rejected"
    )


def test_rejected_candidate_does_not_change_model():
    rng = np.random.RandomState(3)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)
    scores_before = baseline.score(enrollment[:5])

    extreme_candidate = np.full(5, 100.0)
    baseline.offer_candidate(extreme_candidate, round_index=0)
    scores_after = baseline.score(enrollment[:5])

    np.testing.assert_array_almost_equal(scores_before, scores_after, decimal=10)
