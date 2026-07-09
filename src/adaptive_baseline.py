"""
Week 4: a continuously-adapting per-user baseline. This did not exist
before this week -- Weeks 1-3 only fit static models once. This module
implements the absorb low-risk anomalies into the baseline mechanism
described conceptually in the compendium (Sections 6.7/7.5), for the
first time as actual, testable code.

Design: at each adaptation round, a candidate sample is scored against
the CURRENT model. If its score is above an absorption threshold
(computed from the model's own enrollment score distribution, not a
fixed constant -- this makes it adaptive to each individual subject's
own natural score spread), it is folded into the enrollment set and the
model is refit. If not, it is rejected and the enrollment set is
unchanged for that round.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np

from src.models import PerUserModel


@dataclass
class AdaptationRoundResult:
    round_index: int
    candidate_score: float
    absorption_threshold: float
    absorbed: bool


@dataclass
class AdaptiveBaseline:
    """
    Wraps a PerUserModel with a continuously-updating enrollment set.
    """
    algorithm: str = "isolation_forest"
    absorption_percentile: float = 10.0
    enrollment: np.ndarray = field(default=None)
    model: PerUserModel = field(default=None)
    history: List[AdaptationRoundResult] = field(default_factory=list)

    def initialize(self, initial_enrollment: np.ndarray):
        self.enrollment = initial_enrollment.copy()
        self.model = PerUserModel(algorithm=self.algorithm).fit(self.enrollment)
        return self

    def _current_absorption_threshold(self) -> float:
        own_scores = self.model.score(self.enrollment)
        return float(np.percentile(own_scores, self.absorption_percentile))

    def offer_candidate(self, candidate: np.ndarray, round_index: int) -> AdaptationRoundResult:
        """
        candidate: shape (1, n_features) or (n_features,), a single
        repetition being offered for absorption into the baseline.
        """
        candidate = candidate.reshape(1, -1)
        threshold = self._current_absorption_threshold()
        score = float(self.model.score(candidate)[0])
        absorbed = score > threshold

        if absorbed:
            self.enrollment = np.vstack([self.enrollment, candidate])
            self.model = PerUserModel(algorithm=self.algorithm).fit(self.enrollment)

        result = AdaptationRoundResult(
            round_index=round_index,
            candidate_score=score,
            absorption_threshold=threshold,
            absorbed=absorbed,
        )
        self.history.append(result)
        return result

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.score(X)
