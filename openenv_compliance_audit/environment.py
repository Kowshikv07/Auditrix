"""ComplianceAuditEnv — OpenEnv-compatible compliance audit environment.

OpenEnv API
-----------
reset(task_id=None) → AuditObservation
step(action: AuditAction) → StepResult
state() → AuditState

Episode flow
------------
1. Agent calls inspect_record(record_id)          → reveals record fields
2. Agent calls apply_rule(record_id, rule_id)     → runs rule engine; +0.2 if violation found
3. Agent calls flag_violation(record_id, rule_id) → officially flags; +0.5 correct / -0.3 wrong
4. Agent calls mark_compliant(record_id)          → declares no violations; +0.05 if right
5. Agent calls generate_report() or finish()      → episode ends; terminal grader score applied
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .graders import grader_for_difficulty
from .models import (
    ActionType,
    AuditAction,
    AuditObservation,
    AuditReward,
    AuditState,
    RecordInternalState,
    RecordView,
    StepResult,
    ViolationEntry,
)
from .rules import RULES, rule_info
from .tasks import TASKS, list_task_ids


class ComplianceAuditEnv:
    """Real-world compliance audit environment compatible with the OpenEnv spec."""

    benchmark_name = "openenv_compliance_audit"

    def __init__(self, task_id: str = "easy_basic_audit") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list_task_ids()}")
        self._active_task_id = task_id
        self._state: Optional[AuditState] = None
        self._last_action_sig: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def task_ids(self) -> List[str]:
        return list_task_ids()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> AuditObservation:
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
                expected_violations=list(rt.expected_violations),
            )

        self._state = AuditState(
            task_id=task.task_id,
            task_title=task.title,
            objective=task.objective,
            difficulty=task.difficulty,
            active_rule_ids=list(task.active_rule_ids),
            max_steps=task.max_steps,
            records=records,
        )
        self._last_action_sig = None
        return self._build_observation()

    def step(self, action: AuditAction) -> StepResult:
        """Execute one agent action and return (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            grader = grader_for_difficulty(self._state.difficulty)
            task_score = grader.grade(self._state).score
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={
                    "message": "Episode already finished.",
                    "task_score": task_score,
                    "difficulty": self._state.difficulty,
                },
            )

        self._state.step_count += 1
        self._state.last_action_error = None

        # Detect repeated identical action
        sig = (
            action.action_type.value,
            action.record_id,
            action.rule_id,
        )
        is_repeat = sig == self._last_action_sig
        self._last_action_sig = sig

        reward_components: Dict[str, float] = {}
        reward_value = 0.0

        try:
            step_reward, reward_components = self._dispatch(action, is_repeat)
            reward_value += step_reward
        except ValueError as exc:
            self._state.last_action_error = str(exc)
            reward_components["invalid_action"] = 0.0

        # Episode termination
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        terminal_bonus = 0.0
        if action.action_type in (ActionType.GENERATE_REPORT, ActionType.FINISH):
            self._state.done = True
            grader = grader_for_difficulty(self._state.difficulty)
            final_score = grader.grade(self._state).score
            # Report gets a larger terminal bonus than a bare finish
            multiplier = 0.30 if action.action_type == ActionType.GENERATE_REPORT else 0.15
            terminal_bonus = multiplier * final_score
            reward_components["terminal_bonus"] = round(terminal_bonus, 4)
            reward_value += terminal_bonus

        # Clamp to [-1, 1]
        reward_value = max(-1.0, min(1.0, reward_value))

        reward_model = AuditReward(
            value=reward_value,
            components=reward_components,
            rationale=(
                "Shaped reward: +0.2 correct rule application, +0.5 correct violation flag, "
                "-0.3 false positive, -0.05 redundant action, terminal grader bonus on report."
            ),
        )

        task_score = grader_for_difficulty(self._state.difficulty).grade(self._state).score

        action_text = self._action_text(action)
        self._state.action_history.append(action_text)

        return StepResult(
            observation=self._build_observation(),
            reward=reward_model.value,
            done=self._state.done,
            info={
                "reward_model": reward_model.model_dump(),
                "task_score": task_score,
                "difficulty": self._state.difficulty,
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
        self, action: AuditAction, is_repeat: bool
    ) -> Tuple[float, Dict[str, float]]:
        assert self._state is not None
        components: Dict[str, float] = {}

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

        # ── inspect_record ────────────────────────────────────────────
        if action.action_type == ActionType.INSPECT_RECORD:
            if is_repeat or record.inspected:
                self._state.penalties += 0.02
                components["inspect_repeat"] = -0.02
                return -0.02, components
            record.inspected = True
            components["inspect_new"] = 0.06
            return 0.06, components

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
                if is_repeat:
                    self._state.penalties += 0.05
                    components["rule_repeat"] = -0.05
                    return -0.05, components

            # Run the rule engine
            rule = RULES[action.rule_id]
            all_fields = [r.fields for r in self._state.records.values()]
            is_violation = rule.evaluate(record.fields, all_fields)

            if action.rule_id not in record.rules_applied:
                record.rules_applied.append(action.rule_id)

            if is_violation:
                components["rule_hit"] = 0.20
                return 0.20, components
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

            is_true_violation = action.rule_id in record.expected_violations
            record.flagged_violations.append(action.rule_id)

            if is_true_violation:
                components["flag_correct"] = 0.50
                return 0.50, components
            # False positive
            self._state.penalties += 0.15
            components["flag_false_positive"] = -0.30
            return -0.30, components

        # ── mark_compliant ────────────────────────────────────────────
        if action.action_type == ActionType.MARK_COMPLIANT:
            if record.marked_compliant:
                self._state.penalties += 0.02
                components["compliant_repeat"] = -0.02
                return -0.02, components
            record.marked_compliant = True
            has_real_violations = bool(record.expected_violations)
            if not has_real_violations:
                components["compliant_correct"] = 0.05
                return 0.05, components
            # Agent missed violations
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
            # Only expose fields after inspection
            fields = dict(rec.fields) if rec.inspected else {}
            view = RecordView(
                record_id=rec.record_id,
                fields=fields,
                inspected=rec.inspected,
                marked_compliant=rec.marked_compliant,
                flags=list(rec.flagged_violations),
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
        return "(" + ", ".join(parts) + ")"
