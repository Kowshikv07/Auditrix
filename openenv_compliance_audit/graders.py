"""Deterministic graders for the compliance audit environment.

Score formula (per spec §6.3):
    score =
        0.5 * (correct_detected / total_violations)
      + 0.2 * (1 - false_positive_rate)
      + 0.2 * (records_checked / total_records)
      + 0.1 * (1 - steps_used / max_steps)

Difficulty variants slightly reweight the components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

from .models import AuditState


@dataclass(frozen=True)
class GradingBreakdown:
    score: float
    components: Dict[str, float]


class BaseAuditGrader:
    """Compute a normalised [0, 1] score from the terminal episode state."""

    # Weights: (correct_violations, false_positive, coverage, efficiency)
    WEIGHTS: Tuple[float, float, float, float] = (0.5, 0.2, 0.2, 0.1)

    def grade(self, state: AuditState) -> GradingBreakdown:
        w_viol, w_fp, w_cov, w_eff = self.WEIGHTS

        # ── ground truth ──────────────────────────────────────────────────
        true_violations: Set[Tuple[str, str]] = set()
        for rec in state.records.values():
            for rule_id in rec.expected_violations:
                true_violations.add((rec.record_id, rule_id))

        total_violations = len(true_violations)

        # ── agent declarations ─────────────────────────────────────────────
        flagged: Set[Tuple[str, str]] = set()
        for rec in state.records.values():
            for rule_id in rec.flagged_violations:
                flagged.add((rec.record_id, rule_id))

        correct_detected = len(flagged & true_violations)
        false_positives = len(flagged - true_violations)
        total_flagged = len(flagged)

        # ── component scores ───────────────────────────────────────────────
        detection_rate = (
            correct_detected / total_violations if total_violations > 0 else 1.0
        )
        false_positive_rate = (
            false_positives / total_flagged if total_flagged > 0 else 0.0
        )
        records_checked = sum(1 for r in state.records.values() if r.inspected)
        total_records = len(state.records) or 1
        coverage = records_checked / total_records

        efficiency = max(
            0.0, 1.0 - (state.step_count / state.max_steps)
        ) if state.max_steps > 0 else 0.0

        score = (
            w_viol * detection_rate
            + w_fp   * (1.0 - false_positive_rate)
            + w_cov  * coverage
            + w_eff  * efficiency
        )

        components = {
            "detection_rate":      round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "coverage":            round(coverage, 4),
            "efficiency":          round(efficiency, 4),
            "correct_detected":    correct_detected,
            "total_violations":    total_violations,
            "false_positives":     false_positives,
            "records_checked":     records_checked,
            "total_records":       total_records,
        }
        return GradingBreakdown(
            score=max(0.0, min(1.0, score)),
            components=components,
        )


class EasyAuditGrader(BaseAuditGrader):
    """Easy — reward correct detection heavily; coverage matters less."""
    WEIGHTS = (0.55, 0.20, 0.15, 0.10)


class MediumAuditGrader(BaseAuditGrader):
    """Medium — balanced between detection, precision, and coverage."""
    WEIGHTS = (0.50, 0.20, 0.20, 0.10)


class HardAuditGrader(BaseAuditGrader):
    """Hard — full spec weights; efficiency bonus is meaningful."""
    WEIGHTS = (0.50, 0.20, 0.20, 0.10)


def grader_for_difficulty(difficulty: str) -> BaseAuditGrader:
    if difficulty == "easy":
        return EasyAuditGrader()
    if difficulty == "medium":
        return MediumAuditGrader()
    if difficulty == "hard":
        return HardAuditGrader()
    raise ValueError(f"Unknown difficulty: {difficulty!r}")
