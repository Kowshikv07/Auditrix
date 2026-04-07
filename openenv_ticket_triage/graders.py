from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .models import TicketTriageState


@dataclass(frozen=True)
class GradingBreakdown:
    score: float
    components: Dict[str, float]


class BaseGraderAgent:
    """Deterministic task grader that maps final state to a normalized [0, 1] score."""

    def grade(self, state: TicketTriageState) -> GradingBreakdown:
        raise NotImplementedError

    @staticmethod
    def _triage_accuracy(state: TicketTriageState) -> Tuple[float, float]:
        tickets = list(state.tickets.values())
        if not tickets:
            return 0.0, 0.0

        priority_correct = sum(1 for t in tickets if t.priority == t.expected_priority)
        team_correct = sum(1 for t in tickets if t.assigned_team == t.expected_team)
        total = len(tickets)
        return priority_correct / total, team_correct / total

    @staticmethod
    def _compliance_accuracy(state: TicketTriageState) -> float:
        tickets = list(state.tickets.values())
        if not tickets:
            return 0.0

        correct = 0
        for t in tickets:
            if t.requires_compliance and t.compliance_escalated:
                correct += 1
            elif not t.requires_compliance and not t.compliance_escalated:
                correct += 1
        return correct / len(tickets)

    @staticmethod
    def _reply_accuracy(state: TicketTriageState) -> float:
        tickets = list(state.tickets.values())
        if not tickets:
            return 0.0

        correct = 0
        for t in tickets:
            if t.requires_customer_reply and t.requested_customer_reply:
                correct += 1
            elif not t.requires_customer_reply:
                correct += 1
        return correct / len(tickets)

    @staticmethod
    def _resolution_coverage(state: TicketTriageState) -> float:
        tickets = [t for t in state.tickets.values() if t.must_resolve]
        if not tickets:
            return 1.0
        resolved = sum(1 for t in tickets if t.resolved)
        return resolved / len(tickets)

    @staticmethod
    def _efficiency(state: TicketTriageState) -> float:
        if state.max_steps <= 0:
            return 0.0
        # Lower step usage and fewer penalties should score higher.
        step_factor = 1.0 - (state.step_count / state.max_steps)
        penalty_factor = max(0.0, 1.0 - state.penalties)
        return max(0.0, min(1.0, 0.6 * step_factor + 0.4 * penalty_factor))


class EasyGraderAgent(BaseGraderAgent):
    def grade(self, state: TicketTriageState) -> GradingBreakdown:
        pri_acc, team_acc = self._triage_accuracy(state)
        resolution = self._resolution_coverage(state)
        efficiency = self._efficiency(state)

        components = {
            "priority_accuracy": pri_acc,
            "team_accuracy": team_acc,
            "resolution": resolution,
            "efficiency": efficiency,
        }
        score = 0.30 * pri_acc + 0.30 * team_acc + 0.30 * resolution + 0.10 * efficiency
        return GradingBreakdown(score=max(0.0, min(1.0, score)), components=components)


class MediumGraderAgent(BaseGraderAgent):
    def grade(self, state: TicketTriageState) -> GradingBreakdown:
        pri_acc, team_acc = self._triage_accuracy(state)
        resolution = self._resolution_coverage(state)
        reply = self._reply_accuracy(state)
        efficiency = self._efficiency(state)

        components = {
            "priority_accuracy": pri_acc,
            "team_accuracy": team_acc,
            "resolution": resolution,
            "customer_reply_accuracy": reply,
            "efficiency": efficiency,
        }
        score = (
            0.25 * pri_acc
            + 0.25 * team_acc
            + 0.20 * resolution
            + 0.20 * reply
            + 0.10 * efficiency
        )
        return GradingBreakdown(score=max(0.0, min(1.0, score)), components=components)


class HardGraderAgent(BaseGraderAgent):
    def grade(self, state: TicketTriageState) -> GradingBreakdown:
        pri_acc, team_acc = self._triage_accuracy(state)
        resolution = self._resolution_coverage(state)
        compliance = self._compliance_accuracy(state)
        reply = self._reply_accuracy(state)
        efficiency = self._efficiency(state)

        components = {
            "priority_accuracy": pri_acc,
            "team_accuracy": team_acc,
            "resolution": resolution,
            "compliance_accuracy": compliance,
            "customer_reply_accuracy": reply,
            "efficiency": efficiency,
        }
        score = (
            0.20 * pri_acc
            + 0.20 * team_acc
            + 0.20 * resolution
            + 0.25 * compliance
            + 0.10 * reply
            + 0.05 * efficiency
        )
        return GradingBreakdown(score=max(0.0, min(1.0, score)), components=components)


def grader_for_difficulty(difficulty: str) -> BaseGraderAgent:
    if difficulty == "easy":
        return EasyGraderAgent()
    if difficulty == "medium":
        return MediumGraderAgent()
    if difficulty == "hard":
        return HardGraderAgent()
    raise ValueError(f"Unsupported difficulty: {difficulty}")
