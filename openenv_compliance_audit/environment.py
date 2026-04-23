"""ComplianceAuditEnv — OpenEnv-compatible compliance audit environment.

OpenEnv API
-----------
reset(task_id=None, seed=None) → AuditObservation
step(action: AuditAction) → StepResult
state() → AuditState

Episode flow
------------
1. Agent calls inspect_record(record_id)          → reveals record fields
2. Agent calls apply_rule(record_id, rule_id)     → runs rule engine; +0.2 if violation found
3. Agent calls flag_violation(record_id, rule_id) → officially flags; +0.5 correct / -0.3 wrong
4. Agent calls mark_compliant(record_id)           → declares no violations; +0.05 if right
generate_report or finish       → episode ends; terminal grader score applied

Additions (Glacio-inspired improvements)
-------------------------------------------
• Dynamic incident mechanics: EventScheduler injects POLICY_UPDATE / SYSTEM_OUTAGE /
  RECORD_AMENDMENT events mid-episode per (task_id, seed).
• Structured explainability: every apply_rule and flag_violation returns a DecisionTrace
  with full reason_codes and rule_evidence using `evaluate_with_evidence()`.
• Audit confidence report: generate_report accepts and scores an `audit_confidence` section.
• Anti-exploit loop detection: sliding window of last 5 actions; >3 identical
  (action_type, record_id) signals in 5 steps → -0.10 penalty.
• Report consistency check: if the submitted report's flagged_violations contradicts
  the actual flagged state, a proportional terminal penalty is applied.
• event_schedule logged in info["event_schedule"] at reset() for deterministic grading.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .events import EventScheduler
from .graders import grader_for_difficulty
from .models import (
    ActionType,
    AuditAction,
    AuditObservation,
    AuditReward,
    AuditState,
    DecisionTrace,
    EventEntry,
    RecordInternalState,
    RecordView,
    StepResult,
    ViolationEntry,
)
from .rules import RULES, rule_info
from .tasks import TASKS, list_task_ids

# Anti-exploit window size
_LOOP_WINDOW = 5
_LOOP_THRESHOLD = 3   # same sig must appear > this many times in window to trigger penalty
_LOOP_PENALTY = 0.10


class ComplianceAuditEnv:
    """Real-world compliance audit environment compatible with the OpenEnv spec."""

    benchmark_name = "openenv_compliance_audit"

    def __init__(self, task_id: str = "easy_basic_audit") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list_task_ids()}")
        self._active_task_id = task_id
        self._state: Optional[AuditState] = None
        self._last_action_sig: Optional[Tuple] = None
        self._episode_seed: Optional[int] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def task_ids(self) -> List[str]:
        return list_task_ids()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> AuditObservation:
        """Initialise a fresh episode and return the initial observation."""
        if task_id is not None:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id!r}")
            self._active_task_id = task_id

        task = TASKS[self._active_task_id]
        records: Dict[str, RecordInternalState] = {}
        for rt in task.records:
            records[rt.record_id] = RecordInternalState(
                record_id=rt.record_id,
                fields=dict(rt.fields),
                original_fields=dict(rt.fields),  # snapshot before any amendments
                expected_violations=list(rt.expected_violations),
            )

        # Build deterministic event schedule
        resolved_seed = seed if seed is not None else 42
        scheduler = EventScheduler(task_id=self._active_task_id, seed=resolved_seed)
        event_schedule = scheduler.build_schedule()

        self._state = AuditState(
            task_id=task.task_id,
            task_title=task.title,
            objective=task.objective,
            difficulty=task.difficulty,
            active_rule_ids=list(task.active_rule_ids),
            max_steps=task.max_steps,
            records=records,
            event_schedule=event_schedule,
            policy_overrides={},
            recent_action_window=[],
            loop_penalty_applied=0,
        )
        self._last_action_sig = None
        self._episode_seed = resolved_seed
        return self._build_observation()

    def step(self, action: AuditAction) -> StepResult:
        """Execute one agent action and return (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            grader = grader_for_difficulty(self._state.difficulty)
            breakdown = grader.grade(self._state)
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={
                    "message": "Episode already finished.",
                    "task_score": breakdown.score,
                    "failure_mode": breakdown.failure_mode,
                    "difficulty": self._state.difficulty,
                },
            )

        self._state.step_count += 1
        self._state.last_action_error = None

        # ── Apply dynamic events for this step ────────────────────────────
        EventScheduler.apply_events(
            step=self._state.step_count,
            event_schedule=self._state.event_schedule,
            records=self._state.records,  # type: ignore[arg-type]
            policy_overrides=self._state.policy_overrides,
        )

        # ── Anti-exploit: track action signature in sliding window ────────
        sig = f"{action.action_type.value}:{action.record_id or ''}"
        window = self._state.recent_action_window
        window.append(sig)
        if len(window) > _LOOP_WINDOW:
            window.pop(0)

        reward_components: Dict[str, float] = {}
        reward_value = 0.0
        loop_penalty_applied = False

        # Detect looping before dispatch
        if len(window) >= _LOOP_WINDOW:
            same_count = window.count(sig)
            if same_count > _LOOP_THRESHOLD:
                self._state.penalties += _LOOP_PENALTY
                reward_components["loop_penalty"] = -_LOOP_PENALTY
                reward_value -= _LOOP_PENALTY
                loop_penalty_applied = True
                self._state.loop_penalty_applied += 1
                self._state.loop_exploit_signature = sig
                # Clear window to avoid repeated penalisation of the same burst
                self._state.recent_action_window.clear()

        if not loop_penalty_applied:
            try:
                step_reward, reward_components = self._dispatch(action)
                # Streaming reward shaping is handled per-action inside _dispatch().
                # No blanket zero-out here — each action returns the correct value.
                reward_value += step_reward
                self._state.reward_history.append(step_reward)
            except ValueError as exc:
                self._state.last_action_error = str(exc)
                reward_components["invalid_action"] = 0.0
        else:
            # Skip dispatch when loop penalty fired
            pass

        # Episode termination
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        terminal_bonus = 0.0
        report_quality_score = 0.0
        report_quality_components: Dict[str, float] = {}

        if action.action_type in (ActionType.GENERATE_REPORT, ActionType.FINISH):
            self._state.done = True
            grader = grader_for_difficulty(self._state.difficulty)
            breakdown = grader.grade(self._state)
            final_score = breakdown.score

            multiplier = 0.30 if action.action_type == ActionType.GENERATE_REPORT else 0.15
            terminal_bonus = multiplier * final_score
            reward_components["terminal_bonus"] = round(terminal_bonus, 4)
            reward_value += terminal_bonus

            if action.action_type == ActionType.GENERATE_REPORT and action.report is not None:
                report_quality_score, report_quality_components = self._score_report_payload(action.report)
                report_quality_bonus = 0.05 * report_quality_score
                reward_components["report_quality_bonus"] = round(report_quality_bonus, 4)
                reward_value += report_quality_bonus

                # Consistency check: submitted flagged_violations vs actual flagged state
                consistency_penalty = self._report_consistency_penalty(action.report)
                if consistency_penalty > 0:
                    self._state.penalties += consistency_penalty
                    reward_components["report_inconsistency_penalty"] = -round(consistency_penalty, 4)
                    reward_value -= consistency_penalty
                    report_quality_components["consistency_penalty"] = round(consistency_penalty, 4)

                self._state.report_generated = True

        # Clamp to [-1, 1]
        reward_value = max(-1.0, min(1.0, reward_value))

        grader = grader_for_difficulty(self._state.difficulty)
        breakdown = grader.grade(self._state)
        task_score = breakdown.score
        failure_mode = breakdown.failure_mode

        reward_model = AuditReward(
            value=reward_value,
            components=reward_components,
            rationale=(
                "Shaped reward: +0.06 first inspect, +0.2 correct rule application, "
                "+0.5 correct violation flag, -0.3 false positive, -0.05 redundant action, "
                "-0.10 loop exploit detection, terminal grader bonus on report. "
                "Events, loop detection, report consistency check active."
            ),
            failure_mode=failure_mode,
        )

        action_text = self._action_text(action)
        self._state.action_history.append(action_text)

        return StepResult(
            observation=self._build_observation(),
            reward=reward_model.value,
            done=self._state.done,
            info={
                "reward_model": reward_model.model_dump(),
                "task_score": task_score,
                "failure_mode": failure_mode,
                "difficulty": self._state.difficulty,
                "report_quality_score": report_quality_score,
                "report_quality_components": report_quality_components,
                "grading_breakdown": breakdown.components,
                "events_fired_this_step": [
                    e.model_dump()
                    for e in self._state.event_schedule
                    if e.fired and e.trigger_step == self._state.step_count
                ],
            },
        )

    def state(self) -> AuditState:
        """Return a deep copy of the full internal state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        """No-op cleanup hook (required by agent harness convention)."""
        pass

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, action: AuditAction
    ) -> Tuple[float, Dict[str, float]]:
        assert self._state is not None
        components: Dict[str, float] = {}
        self._state.last_decision_trace = None

        # ── prioritize_rules (multi-step planning) ────────────────────
        if action.action_type == ActionType.PRIORITIZE_RULES:
            if not action.rule_priority_order:
                raise ValueError("rule_priority_order is required for prioritize_rules action.")
            # Validate that all active rules are present in the priority order
            provided_rules = set(action.rule_priority_order)
            active_rules = set(self._state.active_rule_ids)
            if provided_rules != active_rules:
                raise ValueError(
                    f"rule_priority_order must contain exactly the active rules. "
                    f"Expected: {sorted(active_rules)}, got: {sorted(provided_rules)}"
                )
            self._state.rule_priority_order = list(action.rule_priority_order)
            self._state.rule_priority_set = True
            components["rule_priority_set"] = 0.0
            return 0.0, components

        # ── generate_report / finish ──────────────────────────────────
        if action.action_type in (ActionType.GENERATE_REPORT, ActionType.FINISH):
            components["pre_terminal"] = 0.0
            return 0.0, components

        # All other actions require a valid record_id
        if not action.record_id:
            raise ValueError("record_id is required for this action.")
        if action.record_id not in self._state.records:
            raise ValueError(f"Unknown record_id: {action.record_id!r}")

        record = self._state.records[action.record_id]

        # ── SYSTEM_OUTAGE guard ───────────────────────────────────────
        if record.system_outage:
            raise ValueError(
                f"Record {action.record_id!r} is currently unavailable due to a system outage. "
                f"Outage ends at step {record.outage_ends_at_step}."
            )

        # ── inspect_record ────────────────────────────────────────────
        if action.action_type == ActionType.INSPECT_RECORD:
            if record.inspected:
                self._state.penalties += 0.02
                components["inspect_repeat"] = -0.02
                return -0.02, components
            record.inspected = True
            # Streaming: tiny shaped reward to encourage record exploration.
            # Small enough to keep long-horizon sparse character; large enough
            # to give untrained models a signal to keep moving forward.
            inspect_reward = 0.01 if self._state.difficulty == "streaming" else 0.06
            components["inspect_new"] = inspect_reward
            return inspect_reward, components

        # Guard: record must be inspected before audit actions
        if not record.inspected:
            raise ValueError(
                f"Record {action.record_id!r} must be inspected before applying rules or flagging."
            )

        # ── apply_rule ────────────────────────────────────────────────
        if action.action_type == ActionType.APPLY_RULE:
            if not action.rule_id:
                raise ValueError("rule_id is required for apply_rule.")
            if action.rule_id not in self._state.active_rule_ids:
                raise ValueError(
                    f"Rule {action.rule_id!r} is not active in this task. "
                    f"Active rules: {self._state.active_rule_ids}"
                )
            if action.rule_id in record.rules_applied:
                self._state.penalties += 0.05
                components["rule_repeat"] = -0.05
                return -0.05, components

            # Run the rule engine with evidence + policy overrides
            rule = RULES[action.rule_id]
            all_fields = [r.fields for r in self._state.records.values()]
            policy_override = self._state.policy_overrides.get(action.rule_id)

            is_violation, reason_codes, evidence = rule.evaluate_with_evidence(
                record.fields, all_fields, policy_override
            )

            if action.rule_id not in record.rules_applied:
                record.rules_applied.append(action.rule_id)

            self._state.last_decision_trace = DecisionTrace(
                action_type=ActionType.APPLY_RULE.value,
                record_id=action.record_id,
                rule_id=action.rule_id,
                outcome="violation_detected" if is_violation else "no_violation",
                reason_codes=reason_codes,
                rule_evidence=evidence,
            )

            if is_violation:
                # apply_rule stays fully sparse in streaming — the model
                # must flag to get any meaningful signal.
                rule_reward = 0.0 if self._state.difficulty == "streaming" else 0.20
                components["rule_hit"] = rule_reward
                return rule_reward, components
            components["rule_no_hit"] = 0.0
            return 0.0, components

        # ── flag_violation ────────────────────────────────────────────
        if action.action_type == ActionType.FLAG_VIOLATION:
            if not action.rule_id:
                raise ValueError("rule_id is required for flag_violation.")
            if action.rule_id not in self._state.active_rule_ids:
                raise ValueError(f"Rule {action.rule_id!r} is not active in this task.")
            if action.rule_id in record.flagged_violations:
                self._state.penalties += 0.05
                components["flag_duplicate"] = -0.05
                return -0.05, components


            # ── Live rule evaluation (source of truth) ────────────────────
            # Evaluate the rule RIGHT NOW using current policy_overrides.
            # This correctly handles two cases:
            #   1. POLICY_UPDATE created a new violation (e.g. overtime 48→40):
            #      The static expected_violations doesn't include this, but the
            #      live evaluation returns True → agent is rewarded correctly.
            #   2. RECORD_AMENDMENT resolved a violation:
            #      The static expected_violations still lists it, but the live
            #      evaluation returns False → flagging is now a false positive.
            rule = RULES[action.rule_id]
            all_fields = [r.fields for r in self._state.records.values()]
            policy_override = self._state.policy_overrides.get(action.rule_id)
            current_eval, _, _ = rule.evaluate_with_evidence(
                record.fields, all_fields, policy_override
            )
            is_true_violation = current_eval

            record.flagged_violations.append(action.rule_id)

            self._state.last_decision_trace = DecisionTrace(
                action_type=ActionType.FLAG_VIOLATION.value,
                record_id=action.record_id,
                rule_id=action.rule_id,
                outcome="flag_correct" if is_true_violation else "flag_false_positive",
                reason_codes=self._reason_codes_for_flag(is_true_violation),
                rule_evidence=self._build_flag_evidence(action.record_id, action.rule_id, is_true_violation),
            )

            if is_true_violation:
                # Streaming: modest positive signal so model learns flagging is
                # the goal — much less than other tasks to keep sparse character.
                flag_reward = 0.05 if self._state.difficulty == "streaming" else 0.50
                components["flag_correct"] = flag_reward
                return flag_reward, components
            self._state.penalties += 0.15
            # Streaming: negative signal on FP so model learns not to guess.
            fp_reward = -0.10 if self._state.difficulty == "streaming" else -0.30
            components["flag_false_positive"] = fp_reward
            return fp_reward, components

        # ── mark_compliant ────────────────────────────────────────────
        if action.action_type == ActionType.MARK_COMPLIANT:
            if record.marked_compliant:
                self._state.penalties += 0.02
                components["compliant_repeat"] = -0.02
                return -0.02, components
            record.marked_compliant = True
            has_real_violations = bool(record.expected_violations)
            if not has_real_violations:
                # mark_compliant stays sparse in streaming.
                compliant_reward = 0.0 if self._state.difficulty == "streaming" else 0.05
                components["compliant_correct"] = compliant_reward
                return compliant_reward, components
            self._state.penalties += 0.10
            components["compliant_wrong"] = -0.10
            return -0.10, components

        raise ValueError(f"Unsupported action_type: {action.action_type!r}")

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> AuditObservation:
        assert self._state is not None

        active_rules = [
            {"rule_id": rid, **{k: v for k, v in r.items() if k != "rule_id"}}
            for rid, r in {
                info["rule_id"]: info
                for info in rule_info()
                if info["rule_id"] in self._state.active_rule_ids
            }.items()
        ]

        visible: List[RecordView] = []
        checked: List[str] = []
        violations_found = []

        for rec in self._state.records.values():
            # Only expose fields after inspection; show outage status always
            fields = dict(rec.fields) if rec.inspected else {}
            if rec.system_outage:
                # Show partial info during outage
                fields = {"_status": "system_unavailable", "_outage_ends_at": rec.outage_ends_at_step}

            view = RecordView(
                record_id=rec.record_id,
                fields=fields,
                inspected=rec.inspected,
                marked_compliant=rec.marked_compliant,
                flags=list(rec.flagged_violations),
                system_outage=rec.system_outage,
            )
            visible.append(view)
            if rec.inspected:
                checked.append(rec.record_id)
            for rule_id in rec.flagged_violations:
                violations_found.append(
                    ViolationEntry(
                        record_id=rec.record_id,
                        rule_id=rule_id,
                        description=RULES[rule_id].description
                        if rule_id in RULES
                        else rule_id,
                    )
                )

        fired_events = [e for e in self._state.event_schedule if e.fired]
        remaining = max(0, self._state.max_steps - self._state.step_count)

        return AuditObservation(
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            objective=self._state.objective,
            available_rules=active_rules,
            step_index=self._state.step_count,
            max_steps=self._state.max_steps,
            remaining_steps=remaining,
            visible_records=visible,
            checked_records=checked,
            violations_found=violations_found,
            action_history=list(self._state.action_history),
            last_action_error=self._state.last_action_error,
            last_decision_trace=self._state.last_decision_trace,
            active_events=fired_events,
            current_policy_overrides=dict(self._state.policy_overrides),
            rule_priority_order=list(self._state.rule_priority_order),
            loop_exploit_signature=self._state.loop_exploit_signature,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_text(action: AuditAction) -> str:
        parts = [f"action_type={action.action_type.value}"]
        if action.record_id:
            parts.append(f"record_id={action.record_id}")
        if action.rule_id:
            parts.append(f"rule_id={action.rule_id}")
        if action.report is not None:
            parts.append("report=provided")
        return "(" + ", ".join(parts) + ")"

    @staticmethod
    def _reason_codes_for_flag(is_true_violation: bool) -> List[str]:
        return ["flag_matches_ground_truth"] if is_true_violation else ["false_positive_flag"]

    def _build_flag_evidence(
        self, record_id: str, rule_id: str, is_true_violation: bool
    ) -> Dict[str, Any]:
        assert self._state is not None
        record = self._state.records[record_id]
        return {
            "rule_id": rule_id,
            "expected_violation": is_true_violation,
            "already_flagged_for_record": list(record.flagged_violations),
        }

    def _report_consistency_penalty(self, report: Dict[str, Any]) -> float:
        """Compute a penalty when the report's flagged_violations contradicts actual state.

        Compares the set of (record_id, rule_id) pairs in the submitted report
        against the set of pairs that were actually flagged via flag_violation.
        Returns a penalty in [0, 0.20].
        """
        assert self._state is not None

        reported_pairs = self._parse_report_pairs(report.get("flagged_violations"))
        actual_pairs: Set[Tuple[str, str]] = set()
        for rec in self._state.records.values():
            for rule_id in rec.flagged_violations:
                actual_pairs.add((rec.record_id, rule_id))

        if not reported_pairs and not actual_pairs:
            return 0.0

        # Symmetric difference: pairs in report but not flagged (or vice versa)
        discrepancy = len(reported_pairs.symmetric_difference(actual_pairs))
        total = max(len(reported_pairs), len(actual_pairs), 1)
        inconsistency_ratio = discrepancy / total
        return round(0.20 * inconsistency_ratio, 4)

    def _score_report_payload(self, report: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Score report completeness/consistency in [0, 1]."""
        assert self._state is not None

        completeness = 0.0
        has_summary = bool(str(report.get("summary", "")).strip())
        has_violations = isinstance(report.get("flagged_violations"), list)
        has_compliant = isinstance(report.get("compliant_records"), list)
        has_recommendations = bool(report.get("recommendations"))
        has_confidence = isinstance(report.get("audit_confidence"), dict)

        completeness += 0.15 if has_summary else 0.0
        completeness += 0.25 if has_violations else 0.0
        completeness += 0.15 if has_compliant else 0.0
        completeness += 0.25 if has_recommendations else 0.0
        completeness += 0.20 if has_confidence else 0.0  # new: confidence section

        expected_pairs: Set[Tuple[str, str]] = set()
        marked_compliant_expected: Set[str] = set()
        for rec in self._state.records.values():
            for rid in rec.flagged_violations:
                expected_pairs.add((rec.record_id, rid))
            if rec.marked_compliant:
                marked_compliant_expected.add(rec.record_id)

        reported_pairs = self._parse_report_pairs(report.get("flagged_violations"))
        reported_compliant = self._parse_record_ids(report.get("compliant_records"))

        pair_f1 = self._f1_score(expected_pairs, reported_pairs)
        compliant_f1 = self._f1_score(marked_compliant_expected, reported_compliant)
        consistency = 0.70 * pair_f1 + 0.30 * compliant_f1

        # Score audit confidence section
        confidence_score = 0.0
        if has_confidence:
            confidence_score = self._score_confidence_section(report["audit_confidence"])

        quality_score = max(
            0.0,
            min(
                1.0,
                0.40 * completeness + 0.45 * consistency + 0.15 * confidence_score
            ),
        )
        components = {
            "completeness": round(completeness, 4),
            "consistency": round(consistency, 4),
            "pair_f1": round(pair_f1, 4),
            "compliant_f1": round(compliant_f1, 4),
            "confidence_score": round(confidence_score, 4),
            "quality_score": round(quality_score, 4),
        }
        return quality_score, components

    def _score_confidence_section(self, confidence: Dict[str, Any]) -> float:
        """Score the audit_confidence section [0, 1].

        Checks:
        - evidence_coverage_ratio accuracy (vs actual records_inspected / total)
        - high_confidence_flags recall: proportion of true violations that the agent
          declares high-confidence (rewards correct certainty)
        - uncertain_flags penalty: if uncertain_flags contains true violations the
          agent already correctly flagged, that's penalised (should be confident)
        """
        assert self._state is not None

        score = 0.0

        # evidence_coverage_ratio accuracy
        actual_coverage = sum(
            1 for r in self._state.records.values() if r.inspected
        ) / max(len(self._state.records), 1)
        stated_coverage = float(confidence.get("evidence_coverage_ratio", 0.0))
        coverage_accuracy = 1.0 - min(1.0, abs(actual_coverage - stated_coverage))
        score += 0.40 * coverage_accuracy

        # Recall of high_confidence_flags on true violations
        true_pairs: Set[str] = set()
        for rec in self._state.records.values():
            for rule_id in rec.expected_violations:
                true_pairs.add(f"{rec.record_id}:{rule_id}")

        hc_flags = set(confidence.get("high_confidence_flags", []))
        if true_pairs:
            hc_recall = len(hc_flags & true_pairs) / len(true_pairs)
        else:
            hc_recall = 1.0
        score += 0.40 * hc_recall

        # Uncertain flags should not contain already-confirmed violations
        uncertain = set(confidence.get("uncertain_flags", []))
        confirmed_flagged: Set[str] = set()
        for rec in self._state.records.values():
            for rule_id in rec.flagged_violations:
                if rule_id in rec.expected_violations:
                    confirmed_flagged.add(f"{rec.record_id}:{rule_id}")
        false_uncertain = len(uncertain & confirmed_flagged)
        uncertainty_penalty = min(0.20, false_uncertain * 0.05)
        score += 0.20 * (1.0 - min(1.0, uncertainty_penalty / 0.20))

        return max(0.0, min(1.0, score))

    @staticmethod
    def _parse_report_pairs(payload: Any) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        if not isinstance(payload, list):
            return pairs
        for item in payload:
            if isinstance(item, dict):
                rec = item.get("record_id")
                rid = item.get("rule_id")
                if isinstance(rec, str) and isinstance(rid, str):
                    pairs.add((rec.strip(), rid.strip()))
            elif isinstance(item, str) and ":" in item:
                rec, rid = item.split(":", 1)
                rec = rec.strip()
                rid = rid.strip()
                if rec and rid:
                    pairs.add((rec, rid))
        return pairs

    @staticmethod
    def _parse_record_ids(payload: Any) -> Set[str]:
        rec_ids: Set[str] = set()
        if not isinstance(payload, list):
            return rec_ids
        for item in payload:
            if isinstance(item, str):
                rec = item.strip()
                if rec:
                    rec_ids.add(rec)
            elif isinstance(item, dict) and isinstance(item.get("record_id"), str):
                rec_ids.add(item["record_id"].strip())
        return rec_ids

    @staticmethod
    def _f1_score(expected: Set[Any], predicted: Set[Any]) -> float:
        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0
        tp = len(expected.intersection(predicted))
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
