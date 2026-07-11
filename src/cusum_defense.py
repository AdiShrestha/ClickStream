"""
Week 5: a residue-feature validation gate, layered IN FRONT of Week
4's existing per-sample absorption check (src/adaptive_baseline.py,
frozen, not modified). Tracks a CUSUM statistic over the residual
between each candidate's score and a FIXED reference score computed
once from the TRUE original (pre-poisoning) enrollment set.

The reference score is never recomputed from the current, possibly-
already-drifted enrollment -- if it were, slow poisoning could drag the
reference along with it, and the defense would never trigger no matter
how long the attack ran. Anchoring to the original baseline is what
makes this a defense against gradual drift specifically, not just
another per-sample outlier check with extra steps.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np

from src.adaptive_baseline import AdaptiveBaseline, AdaptationRoundResult
from src.models import PerUserModel


@dataclass
class DefendedAdaptiveBaseline(AdaptiveBaseline):
    cusum_k: float = 0.0   # slack parameter -- allowed "free" noise per round
    cusum_h: float = 0.0   # alarm threshold -- calibrated on benign data only (Section 4)
    _reference_score: float = field(default=None, repr=False)
    _cusum_state: float = field(default=0.0, repr=False)
    cusum_history: List[float] = field(default_factory=list)
    defense_triggered_rounds: List[int] = field(default_factory=list)

    def initialize(self, initial_enrollment: np.ndarray):
        super().initialize(initial_enrollment)
        original_scores = self.model.score(initial_enrollment)
        self._reference_score = float(np.median(original_scores))
        self._cusum_state = 0.0
        return self

    def offer_candidate(self, candidate: np.ndarray, round_index: int) -> AdaptationRoundResult:
        candidate = candidate.reshape(1, -1)
        candidate_score = float(self.model.score(candidate)[0])

        # Residual: how much LESS normal-looking is this candidate than
        # the TRUE original baseline (not the current, possibly-drifted one).
        residual = self._reference_score - candidate_score
        self._cusum_state = max(0.0, self._cusum_state + residual - self.cusum_k)
        self.cusum_history.append(self._cusum_state)

        defense_triggered = self._cusum_state > self.cusum_h
        if defense_triggered:
            self.defense_triggered_rounds.append(round_index)

        threshold = self._current_absorption_threshold()
        passes_per_sample_check = candidate_score > threshold
        # BOTH gates must pass -- the CUSUM check does not replace the
        # existing per-sample check, it adds a second, sequential one.
        absorbed = passes_per_sample_check and not defense_triggered

        if absorbed:
            self.enrollment = np.vstack([self.enrollment, candidate])
            if len(self.enrollment) > self.max_enrollment_size:
                self.enrollment = self.enrollment[-self.max_enrollment_size:]
            self.model = PerUserModel(algorithm=self.algorithm).fit(self.enrollment)

        result = AdaptationRoundResult(
            round_index=round_index,
            candidate_score=candidate_score,
            absorption_threshold=threshold,
            absorbed=absorbed,
        )
        self.history.append(result)
        return result
