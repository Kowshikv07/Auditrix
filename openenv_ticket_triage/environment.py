from __future__ import annotations

from typing import Dict, Optional, Tuple

from .graders import grader_for_difficulty
from .models import (
    ActionType,
    StepResult,
    Team,
    TicketInternalState,
    TicketPriority,
    TicketTriageAction,
    TicketTriageObservation,
    TicketTriageReward,
    TicketTriageState,
    TicketView,
)
from .tasks import TASKS, list_task_ids


class TicketTriageEnv:
    """Real-world simulation for customer support ticket triage and incident routing."""

    benchmark_name = "openenv_ticket_triage"

    def __init__(self, task_id: str = "easy_refund_priority") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self._active_task_id = task_id
        self._state: Optional[TicketTriageState] = None
        self._last_action_signature: Optional[Tuple[str, Optional[str], Optional[str]]] = None

    @property
    def task_ids(self) -> list[str]:
        return list_task_ids()

    def reset(self, task_id: Optional[str] = None) -> TicketTriageObservation:
        if task_id is not None:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id}")
            self._active_task_id = task_id

        task = TASKS[self._active_task_id]
        tickets: Dict[str, TicketInternalState] = {}
        for t in task.tickets:
            tickets[t.ticket_id] = TicketInternalState(
                ticket_id=t.ticket_id,
                subject=t.subject,
                body=t.body,
                customer_tier=t.customer_tier,
                age_hours=t.age_hours,
                expected_priority=t.expected_priority,
                expected_team=t.expected_team,
                requires_customer_reply=t.requires_customer_reply,
                requires_compliance=t.requires_compliance,
                must_resolve=t.must_resolve,
            )

        self._state = TicketTriageState(
            task_id=task.task_id,
            task_title=task.title,
            objective=task.objective,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
            tickets=tickets,
        )
        self._last_action_signature = None
        return self._to_observation(progress_score=0.0)

    def step(self, action: TicketTriageAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._state.done:
            return StepResult(
                observation=self._to_observation(progress_score=self._final_score()),
                reward=0.0,
                done=True,
                info={"message": "Episode already finished"},
            )

        self._state.step_count += 1
        self._state.last_action_error = None

        reward_components: Dict[str, float] = {}
        reward_value = 0.0

        try:
            increment, reward_components = self._apply_action(action)
            reward_value += increment
        except ValueError as exc:
            self._state.last_action_error = str(exc)
            reward_components["invalid_action"] = 0.0

        signature = (action.action_type.value, action.ticket_id, action.value)
        if signature == self._last_action_signature:
            self._state.penalties += 0.08
            reward_components["repeat_penalty"] = -0.08
            reward_value -= 0.08
        self._last_action_signature = signature

        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        if action.action_type == ActionType.FINISH:
            self._state.done = True
            final_score = self._final_score()
            reward_components["final_score_bonus"] = 0.30 * final_score
            reward_value += 0.30 * final_score

        # Keep reward in [0, 1] while preserving penalties through subtraction.
        reward_value = max(0.0, min(1.0, reward_value))

        progress = self._final_score() if self._state.done else self._progress_proxy()
        reward_model = TicketTriageReward(
            value=reward_value,
            components=reward_components,
            rationale="Shaped reward with partial credit, efficiency penalties, and final deterministic score.",
        )
        obs = self._to_observation(progress_score=progress)

        self._state.action_history.append(self._action_to_text(action))
        return StepResult(
            observation=obs,
            reward=reward_model.value,
            done=self._state.done,
            info={
                "reward_model": reward_model.model_dump(),
                "task_score": self._final_score(),
                "difficulty": self._state.difficulty,
            },
        )

    def state(self) -> TicketTriageState:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._state.model_copy(deep=True)

    def _apply_action(self, action: TicketTriageAction) -> Tuple[float, Dict[str, float]]:
        assert self._state is not None
        components: Dict[str, float] = {}

        if action.action_type == ActionType.FINISH:
            components["finish"] = 0.02
            return 0.02, components

        if action.ticket_id is None:
            raise ValueError("ticket_id is required for this action")
        if action.ticket_id not in self._state.tickets:
            raise ValueError(f"Unknown ticket_id: {action.ticket_id}")

        ticket = self._state.tickets[action.ticket_id]

        if action.action_type == ActionType.INSPECT_TICKET:
            if ticket.inspected:
                components["inspect_repeat"] = -0.02
                self._state.penalties += 0.02
                return -0.02, components
            ticket.inspected = True
            components["inspect_new"] = 0.06
            return 0.06, components

        if action.action_type == ActionType.SET_PRIORITY:
            if not action.value:
                raise ValueError("value is required for set_priority")
            value = action.value.lower()
            if value not in {"low", "medium", "high", "critical"}:
                raise ValueError("priority must be one of low|medium|high|critical")
            ticket.priority = TicketPriority(value)
            if ticket.priority == ticket.expected_priority:
                components["priority_correct"] = 0.12
                return 0.12, components
            components["priority_incorrect"] = 0.02
            self._state.penalties += 0.03
            return 0.02, components

        if action.action_type == ActionType.ASSIGN_TEAM:
            if not action.value:
                raise ValueError("value is required for assign_team")
            value = action.value.lower()
            allowed = {"billing", "support", "product", "security", "fraud", "privacy"}
            if value not in allowed:
                raise ValueError("team must be one of billing|support|product|security|fraud|privacy")
            ticket.assigned_team = Team(value)
            if ticket.assigned_team == ticket.expected_team:
                components["team_correct"] = 0.12
                return 0.12, components
            components["team_incorrect"] = 0.02
            self._state.penalties += 0.03
            return 0.02, components

        if action.action_type == ActionType.REQUEST_CUSTOMER_REPLY:
            ticket.requested_customer_reply = True
            if ticket.requires_customer_reply:
                components["reply_required"] = 0.08
                return 0.08, components
            components["reply_unnecessary"] = 0.01
            self._state.penalties += 0.01
            return 0.01, components

        if action.action_type == ActionType.ESCALATE_COMPLIANCE:
            ticket.compliance_escalated = True
            if ticket.requires_compliance:
                components["compliance_correct"] = 0.14
                return 0.14, components
            components["compliance_unnecessary"] = 0.0
            self._state.penalties += 0.04
            return 0.0, components

        if action.action_type == ActionType.ADD_INTERNAL_NOTE:
            if not action.value:
                raise ValueError("value is required for add_internal_note")
            ticket.internal_notes.append(action.value)
            components["note_added"] = 0.03
            return 0.03, components

        if action.action_type == ActionType.RESOLVE_TICKET:
            ticket.resolved = True
            correct = (
                ticket.priority == ticket.expected_priority
                and ticket.assigned_team == ticket.expected_team
                and ((not ticket.requires_compliance) or ticket.compliance_escalated)
            )
            if correct:
                components["resolve_correct"] = 0.18
                return 0.18, components
            components["resolve_partial"] = 0.03
            self._state.penalties += 0.05
            return 0.03, components

        raise ValueError(f"Unsupported action: {action.action_type}")

    def _progress_proxy(self) -> float:
        assert self._state is not None
        ticket_count = len(self._state.tickets)
        if ticket_count == 0:
            return 0.0

        progress = 0.0
        for ticket in self._state.tickets.values():
            if ticket.inspected:
                progress += 0.10
            if ticket.priority == ticket.expected_priority:
                progress += 0.20
            if ticket.assigned_team == ticket.expected_team:
                progress += 0.20
            if ticket.requires_customer_reply and ticket.requested_customer_reply:
                progress += 0.10
            if ticket.requires_compliance and ticket.compliance_escalated:
                progress += 0.20
            if ticket.must_resolve and ticket.resolved:
                progress += 0.20

        normalized = progress / ticket_count
        normalized -= min(0.3, self._state.penalties * 0.2)
        return max(0.0, min(1.0, normalized))

    def _final_score(self) -> float:
        assert self._state is not None
        grader = grader_for_difficulty(self._state.difficulty)
        return grader.grade(self._state).score

    def _to_observation(self, progress_score: float) -> TicketTriageObservation:
        assert self._state is not None

        visible = [
            TicketView(
                ticket_id=t.ticket_id,
                subject=t.subject,
                customer_tier=t.customer_tier,
                age_hours=t.age_hours,
                inspected=t.inspected,
                priority=t.priority,
                assigned_team=t.assigned_team,
                requested_customer_reply=t.requested_customer_reply,
                compliance_escalated=t.compliance_escalated,
                resolved=t.resolved,
            )
            for t in self._state.tickets.values()
        ]
        return TicketTriageObservation(
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            objective=self._state.objective,
            step_index=self._state.step_count,
            max_steps=self._state.max_steps,
            progress_score=max(0.0, min(1.0, progress_score)),
            visible_tickets=visible,
            action_history=list(self._state.action_history),
            last_action_error=self._state.last_action_error,
        )

    @staticmethod
    def _action_to_text(action: TicketTriageAction) -> str:
        return f"{action.action_type.value}(ticket_id={action.ticket_id}, value={action.value})"
