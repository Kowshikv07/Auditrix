"""Deterministic graders for the compliance audit environment.

Score formula (per spec §6.3):
    score =
        0.5 * (correct_detected / total_violations)
      + 0.2 * (1 - false_positive_rate)
      + 0.2 * (records_checked / total_records)
      + 0.1 * (1 - steps_used / max_steps)

Difficulty variants slightly reweight the components.

Failure Mode Taxonomy
---------------------
Every grader.grade() call returns a GradingBreakdown that includes a dominant
failure_mode drawn from: false_positive, missed_violation, low_coverage,
inefficiency, loop_exploit, report_inconsistency, none.

This powers the [END] structured log line in inference.py and the per-seed
variance reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

from .models import AuditState, FailureMode
from .rules import RULES, VIOLATION_SEVERITY, SEVERITY_WEIGHTS


@dataclass(frozen=True)
class GradingBreakdown:
    score: float
    components: Dict[str, float]
    failure_mode: FailureMode = "none"


def _classify_failure_mode(
    state: AuditState,
    detection_rate: float,
    false_positive_rate: float,
    coverage: float,
    efficiency: float,
) -> FailureMode:
    """Classify the dominant failure mode for post-hoc analysis.

    Priority order: loop_exploit > report_inconsistency > false_positive
    > missed_violation > low_coverage > inefficiency > none.
    """
    # Loop exploit: more than 3 loop penalties applied
    if state.loop_penalty_applied >= 3:
        return "loop_exploit"

    # Report inconsistency: environment tracked a consistency penalty
    if state.penalties > 0 and state.report_generated:
        # Heuristic: large penalty with report generated signals inconsistency
        if state.penalties >= 0.20:
            return "report_inconsistency"

    # False positives dominate
    if false_positive_rate > 0.3:
        return "false_positive"

    # Many missed violations
    if detection_rate < 0.5:
        return "missed_violation"

    # Low coverage
    if coverage < 0.5:
        return "low_coverage"

    # Inefficiency
    if efficiency < 0.1 and state.max_steps > 0:
        return "inefficiency"

    return "none"


class BaseAuditGrader:
    """Compute a normalized [0, 1] score from the terminal episode state."""

    # Weights: (severity_detection, precision, coverage, efficiency, prioritization)
    WEIGHTS: Tuple[float, float, float, float, float] = (0.45, 0.20, 0.15, 0.10, 0.10)

    def _live_true_violations(self, state: AuditState) -> Set[Tuple[str, str]]:
        """Compute the ground-truth violation set by evaluating rules live.

        Unlike the static `expected_violations` baked in at task creation,
        this respects mid-episode changes:
          - POLICY_UPDATE: lower overtime threshold creates new violations
          - RECORD_AMENDMENT: corrected fields resolve old violations
        """
        all_fields = [r.fields for r in state.records.values()]
        true_violations: Set[Tuple[str, str]] = set()
        active = [r for r in state.active_rule_ids if r not in state.suspended_rule_ids]
        for rec in state.records.values():
            for rule_id in active:
                policy_override = state.policy_overrides.get(rule_id)
                rule = RULES.get(rule_id)
                if rule is None:
                    continue
                is_viol, _, _ = rule.evaluate_with_evidence(
                    rec.fields, all_fields, policy_override
                )
                if is_viol:
                    true_violations.add((rec.record_id, rule_id))
        return true_violations

    def _severity_weighted_detection(self, flagged: Set, true_violations: Set) -> float:
        """Detection rate weighted by violation severity.

        A missed GDPR violation (weight=2.0) hurts more than a missed intern-hours
        violation (weight=0.5). Agents that triage correctly score higher.
        """
        if not true_violations:
            return 1.0
        total_weight = sum(
            SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY.get(rid, "medium"), 1.0)
            for (_, rid) in true_violations
        )
        correct_weight = sum(
            SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY.get(rid, "medium"), 1.0)
            for (rec_id, rid) in (flagged & true_violations)
        )
        return correct_weight / total_weight if total_weight > 0 else 1.0

    def _warning_credit(self, state: AuditState, true_violations: Set) -> float:
        """Credit for correctly abstaining on warning-tier cases.

        For each (record_id, rule_id) pair where apply_rule returned 'warning':
        - If the agent did NOT flag it (correct restraint) → +0.05 credit per instance
        - The total warning_credit is capped at 0.10
        """
        if not state.warning_calls:
            return 0.0
        credit = 0.0
        for key in state.warning_calls:
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            rec_id, rule_id = parts
            rec = state.records.get(rec_id)
            if rec is None:
                continue
            # If NOT flagged → agent correctly abstained on ambiguous case
            if rule_id not in rec.flagged_violations:
                credit += 0.05
        return min(0.10, credit)

    def _prioritization_score(self, state: AuditState, true_violations: Set) -> float:
        """Score how well the agent's flagging order reflects violation severity.

        Computes the correlation between flag order and severity weight.
        Returns a value in [0, 1].
        """
        if not state.rule_priority_order or not true_violations:
            return 0.5  # neutral if no priority declared

        # Build expected order from severity (highest first)
        priority_order = state.rule_priority_order
        n = len(priority_order)
        if n == 0:
            return 0.5

        # Score: proportion of correctly ordered critical-before-low pairs
        correct_pairs = 0
        total_pairs = 0
        for i, r1 in enumerate(priority_order):
            for r2 in priority_order[i + 1:]:
                sev1 = SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY.get(r1, "medium"), 1.0)
                sev2 = SEVERITY_WEIGHTS.get(VIOLATION_SEVERITY.get(r2, "medium"), 1.0)
                if sev1 != sev2:
                    total_pairs += 1
                    if sev1 > sev2:  # r1 higher severity listed before r2 lower severity
                        correct_pairs += 1
        if total_pairs == 0:
            return 1.0  # all same severity — any order is fine
        return correct_pairs / total_pairs

    def grade(self, state: AuditState) -> GradingBreakdown:
        w_det, w_fp, w_cov, w_eff, w_pri = self.WEIGHTS

        # ── ground truth (live — respects POLICY_UPDATE & RECORD_AMENDMENT) ─
        true_violations = self._live_true_violations(state)
        total_violations = len(true_violations)

        # ── agent declarations ──────────────────────────────────────
        flagged: Set[Tuple[str, str]] = set()
        for rec in state.records.values():
            for rule_id in rec.flagged_violations:
                flagged.add((rec.record_id, rule_id))

        correct_detected = len(flagged & true_violations)
        false_positives = len(flagged - true_violations)
        total_flagged = len(flagged)

        # ── component scores ───────────────────────────────────────
        severity_detection = self._severity_weighted_detection(flagged, true_violations)
        false_positive_rate = (
            false_positives / total_flagged if total_flagged > 0 else 0.0
        )
        records_checked = sum(1 for r in state.records.values() if r.inspected)
        total_records = len(state.records) or 1
        coverage = records_checked / total_records

        efficiency = max(
            0.0, 1.0 - (state.step_count / state.max_steps)
        ) if state.max_steps > 0 else 0.0

        warning_credit = self._warning_credit(state, true_violations)
        prioritization = self._prioritization_score(state, true_violations)

        # Anti-exploit: loop penalty deduction
        loop_deduction = min(0.10, state.loop_penalty_applied * 0.02)

        score = (
            w_det * severity_detection
            + w_fp  * (1.0 - false_positive_rate)
            + w_cov * coverage
            + w_eff * efficiency
            + w_pri * prioritization
            + 0.05  * warning_credit  # bonus on top
            - loop_deduction
        )

        # Coverage floor: if < 50% inspected at report time, cap terminal contribution
        if coverage < 0.5:
            score = min(score, 0.5)

        # Backward-compat: keep detection_rate as simple ratio for logging
        detection_rate = correct_detected / total_violations if total_violations > 0 else 1.0

        # Failure mode classification
        failure_mode = _classify_failure_mode(
            state, detection_rate, false_positive_rate, coverage, efficiency
        )

        components = {
            "severity_detection":   round(severity_detection, 4),
            "detection_rate":       round(detection_rate, 4),
            "false_positive_rate":  round(false_positive_rate, 4),
            "coverage":             round(coverage, 4),
            "efficiency":           round(efficiency, 4),
            "prioritization_score": round(prioritization, 4),
            "warning_credit":       round(warning_credit, 4),
            "correct_detected":     correct_detected,
            "total_violations":     total_violations,
            "false_positives":      false_positives,
            "records_checked":      records_checked,
            "total_records":        total_records,
            "loop_deduction":       round(loop_deduction, 4),
        }
        return GradingBreakdown(
            score=max(0.0, min(1.0, score)),
            components=components,
            failure_mode=failure_mode,
        )


class EasyAuditGrader(BaseAuditGrader):
    """Easy — reward correct detection heavily; coverage matters less."""
    WEIGHTS = (0.50, 0.20, 0.15, 0.10, 0.05)


class MediumAuditGrader(BaseAuditGrader):
    """Medium — balanced between detection, precision, and coverage."""
    WEIGHTS = (0.45, 0.20, 0.20, 0.10, 0.05)


class HardAuditGrader(BaseAuditGrader):
    """Hard — full spec weights; efficiency bonus is meaningful."""
    WEIGHTS = (0.45, 0.20, 0.15, 0.10, 0.10)


class ExtremeAuditGrader(BaseAuditGrader):
    """Extreme — false-positive-free detection is paramount.

    Grader formula adjustments:
      - Detection rate weight raised (catching all violations matters most)
      - False-positive penalty raised (any FP is heavily penalised)
      - Coverage floor enforced at 60% (not 50%)
      - Loop deduction scales more aggressively (0.03 per penalty vs 0.02)

    Threshold checks:
      - score <= 0.50 if false_positive_rate > 0 (any FP hard-caps partial credit)
      - score <= 0.30 if coverage < 0.40 (inspecting < 40% records → low score)

    The grader uses expected_violations as the ground truth — including those
    resolved by RECORD_AMENDMENT events (which the environment applies to the
    live `fields` dict, making the rule return False after the amendment fires).
    The agent is not penalised for NOT flagging a violation that was resolved
    by a mid-episode RECORD_AMENDMENT.
    """

    WEIGHTS = (0.50, 0.25, 0.13, 0.05, 0.07)  # Detection + FP precision are paramount

    def grade(self, state: AuditState) -> GradingBreakdown:
        w_det, w_fp, w_cov, w_eff, w_pri = self.WEIGHTS

        # Live ground truth
        true_violations = self._live_true_violations(state)
        total_violations = len(true_violations)

        flagged: Set[Tuple[str, str]] = set()
        for rec in state.records.values():
            for rule_id in rec.flagged_violations:
                flagged.add((rec.record_id, rule_id))

        correct_detected = len(flagged & true_violations)
        false_positives = len(flagged - true_violations)
        total_flagged = len(flagged)

        severity_detection = self._severity_weighted_detection(flagged, true_violations)
        detection_rate = correct_detected / total_violations if total_violations > 0 else 1.0
        false_positive_rate = false_positives / total_flagged if total_flagged > 0 else 0.0
        records_checked = sum(1 for r in state.records.values() if r.inspected)
        total_records = len(state.records) or 1
        coverage = records_checked / total_records
        efficiency = max(0.0, 1.0 - (state.step_count / state.max_steps)) if state.max_steps > 0 else 0.0
        prioritization = self._prioritization_score(state, true_violations)
        warning_credit = self._warning_credit(state, true_violations)

        # More aggressive loop deduction for extreme difficulty
        loop_deduction = min(0.15, state.loop_penalty_applied * 0.03)

        score = (
            w_det * severity_detection
            + w_fp  * (1.0 - false_positive_rate)
            + w_cov * coverage
            + w_eff * efficiency
            + w_pri * prioritization
            + 0.05  * warning_credit
            - loop_deduction
        )

        # Hard cap: any false positive limits score to 0.50
        if false_positives > 0:
            score = min(score, 0.50)

        # Hard cap: < 40% coverage → score ≤ 0.30
        if coverage < 0.40:
            score = min(score, 0.30)

        # Coverage floor at 60% (not 50%)
        if coverage < 0.60:
            score = min(score, 0.65)

        failure_mode = _classify_failure_mode(
            state, detection_rate, false_positive_rate, coverage, efficiency
        )

        components = {
            "severity_detection":   round(severity_detection, 4),
            "detection_rate":       round(detection_rate, 4),
            "false_positive_rate":  round(false_positive_rate, 4),
            "coverage":             round(coverage, 4),
            "efficiency":           round(efficiency, 4),
            "prioritization_score": round(prioritization, 4),
            "warning_credit":       round(warning_credit, 4),
            "correct_detected":     correct_detected,
            "total_violations":     total_violations,
            "false_positives":      false_positives,
            "records_checked":      records_checked,
            "total_records":        total_records,
            "loop_deduction":       round(loop_deduction, 4),
        }
        return GradingBreakdown(
            score=max(0.0, min(1.0, score)),
            components=components,
            failure_mode=failure_mode,
        )

class StreamingAuditGrader(BaseAuditGrader):
    """Streaming — long-horizon sparse reward environment.

    Grader formula adjustments:
      - Detection rate weight prioritised (agent must find violations with sparse feedback)
      - False-positive penalty less harsh than extreme (cap at 0.60 instead of 0.50)
        because sparse rewards make it harder to avoid false positives
      - Coverage floor at 30% (more lenient than extreme's 40%)
      - Loop deduction scales moderately (0.025 per penalty)
      - Sparse reward: RECORD_AMENDMENT events mid-episode remove violations from
        true_violations set. Agent is not penalized for missing a violation that was
        resolved by a corrective amendment.

    Threshold checks:
      - score <= 0.60 if false_positive_rate > 0.15 (FP precision matters but not absolute)
      - score <= 0.25 if coverage < 0.30 (< 30% inspected → very low score)

    Purpose: Reward strategic planning and sampling in a 1000-record, 300-step budget
    where the agent must decide which records and rules to prioritize. Delayed rewards
    via RECORD_AMENDMENT events test multi-step planning.
    """

    WEIGHTS = (0.50, 0.25, 0.15, 0.05, 0.05)

    def grade(self, state: AuditState) -> GradingBreakdown:
        w_det, w_fp, w_cov, w_eff, w_pri = self.WEIGHTS

        # Live ground truth — correctly handles both POLICY_UPDATE and RECORD_AMENDMENT.
        true_violations = self._live_true_violations(state)
        total_violations = len(true_violations)

        flagged: Set[Tuple[str, str]] = set()
        for rec in state.records.values():
            for rule_id in rec.flagged_violations:
                flagged.add((rec.record_id, rule_id))

        correct_detected = len(flagged & true_violations)
        false_positives = len(flagged - true_violations)
        total_flagged = len(flagged)

        detection_rate = correct_detected / total_violations if total_violations > 0 else 1.0
        false_positive_rate = false_positives / total_flagged if total_flagged > 0 else 0.0
        records_checked = sum(1 for r in state.records.values() if r.inspected)
        total_records = len(state.records) or 1
        coverage = records_checked / total_records
        efficiency = max(0.0, 1.0 - (state.step_count / state.max_steps)) if state.max_steps > 0 else 0.0
        severity_detection = self._severity_weighted_detection(flagged, true_violations)
        prioritization = self._prioritization_score(state, true_violations)
        warning_credit = self._warning_credit(state, true_violations)

        loop_deduction = min(0.10, state.loop_penalty_applied * 0.025)

        score = (
            w_det * severity_detection
            + w_fp  * (1.0 - false_positive_rate)
            + w_cov * coverage
            + w_eff * efficiency
            + w_pri * prioritization
            + 0.05  * warning_credit
            - loop_deduction
        )

        if false_positive_rate > 0.15:
            score = min(score, 0.60)

        if coverage < 0.30:
            score = min(score, 0.25)

        if coverage < 0.50:
            score = min(score, 0.50)

        failure_mode = _classify_failure_mode(
            state, detection_rate, false_positive_rate, coverage, efficiency
        )

        components = {
            "severity_detection":   round(severity_detection, 4),
            "detection_rate":       round(detection_rate, 4),
            "false_positive_rate":  round(false_positive_rate, 4),
            "coverage":             round(coverage, 4),
            "efficiency":           round(efficiency, 4),
            "prioritization_score": round(prioritization, 4),
            "warning_credit":       round(warning_credit, 4),
            "correct_detected":     correct_detected,
            "total_violations":     total_violations,
            "false_positives":      false_positives,
            "records_checked":      records_checked,
            "total_records":        total_records,
            "loop_deduction":       round(loop_deduction, 4),
        }
        return GradingBreakdown(
            score=max(0.0, min(1.0, score)),
            components=components,
            failure_mode=failure_mode,
        )


def grader_for_difficulty(difficulty: str) -> BaseAuditGrader:
    if difficulty == "easy":
        return EasyAuditGrader()
    if difficulty == "medium":
        return MediumAuditGrader()
    if difficulty == "hard":
        return HardAuditGrader()
    if difficulty == "extreme":
        return ExtremeAuditGrader()
    if difficulty == "streaming":
        return StreamingAuditGrader()
    raise ValueError(f"Unknown difficulty: {difficulty!r}")
