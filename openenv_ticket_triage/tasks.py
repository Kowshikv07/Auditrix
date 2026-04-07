from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from .models import Team, TicketPriority


@dataclass(frozen=True)
class TicketTruth:
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    age_hours: int
    expected_priority: TicketPriority
    expected_team: Team
    requires_customer_reply: bool = False
    requires_compliance: bool = False
    must_resolve: bool = True


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int
    tickets: List[TicketTruth]


TASKS: Dict[str, TaskDefinition] = {
    "easy_refund_priority": TaskDefinition(
        task_id="easy_refund_priority",
        title="Urgent Refund Ticket",
        difficulty="easy",
        objective=(
            "Correctly triage and resolve a single urgent refund ticket by inspecting the ticket, "
            "setting a correct priority, assigning the right team, and resolving it."
        ),
        max_steps=10,
        tickets=[
            TicketTruth(
                ticket_id="T-1001",
                subject="Double charge on invoice INV-4421",
                body=(
                    "I was charged twice for this month. I need this fixed today because my card "
                    "is near limit."
                ),
                customer_tier="pro",
                age_hours=9,
                expected_priority=TicketPriority.HIGH,
                expected_team=Team.BILLING,
                requires_customer_reply=False,
                requires_compliance=False,
            )
        ],
    ),
    "medium_mixed_queue": TaskDefinition(
        task_id="medium_mixed_queue",
        title="Mixed Support Queue",
        difficulty="medium",
        objective=(
            "Triage a realistic queue of 3 tickets. Route each ticket to the right team with the "
            "right priority, request customer follow-up for the phishing report, and resolve at "
            "least two tickets correctly."
        ),
        max_steps=18,
        tickets=[
            TicketTruth(
                ticket_id="T-2001",
                subject="SSO login outage for APAC region",
                body="All enterprise users in APAC cannot log in after the release.",
                customer_tier="enterprise",
                age_hours=2,
                expected_priority=TicketPriority.CRITICAL,
                expected_team=Team.SUPPORT,
            ),
            TicketTruth(
                ticket_id="T-2002",
                subject="Need CSV export for weekly metrics",
                body="Could you add CSV export in analytics? No rush.",
                customer_tier="free",
                age_hours=36,
                expected_priority=TicketPriority.LOW,
                expected_team=Team.PRODUCT,
            ),
            TicketTruth(
                ticket_id="T-2003",
                subject="Possible phishing from reset-password email",
                body="Received suspicious password reset email; can share full headers.",
                customer_tier="pro",
                age_hours=5,
                expected_priority=TicketPriority.MEDIUM,
                expected_team=Team.SECURITY,
                requires_customer_reply=True,
                must_resolve=False,
            ),
        ],
    ),
    "hard_regulated_incident": TaskDefinition(
        task_id="hard_regulated_incident",
        title="Regulated Incident Triage",
        difficulty="hard",
        objective=(
            "Handle a high-risk queue with compliance constraints: route all tickets correctly, "
            "escalate required compliance cases, request customer reply where needed, and resolve "
            "all resolvable tickets with minimal wasted actions."
        ),
        max_steps=24,
        tickets=[
            TicketTruth(
                ticket_id="T-3001",
                subject="Customer pasted full card number in chat",
                body=(
                    "I sent my full card details in chat by mistake, please update my payment and "
                    "remove old messages."
                ),
                customer_tier="pro",
                age_hours=1,
                expected_priority=TicketPriority.HIGH,
                expected_team=Team.BILLING,
                requires_customer_reply=True,
                requires_compliance=True,
            ),
            TicketTruth(
                ticket_id="T-3002",
                subject="Unauthorized purchases flagged by bank",
                body="Three overnight purchases look fraudulent and card was not present.",
                customer_tier="enterprise",
                age_hours=3,
                expected_priority=TicketPriority.CRITICAL,
                expected_team=Team.FRAUD,
                requires_compliance=True,
            ),
            TicketTruth(
                ticket_id="T-3003",
                subject="Billed after downgrading plan",
                body="I downgraded to free last week but was charged for pro.",
                customer_tier="free",
                age_hours=20,
                expected_priority=TicketPriority.MEDIUM,
                expected_team=Team.BILLING,
            ),
            TicketTruth(
                ticket_id="T-3004",
                subject="GDPR delete request for all account records",
                body="Please delete all my personal data and confirm completion timeline.",
                customer_tier="pro",
                age_hours=11,
                expected_priority=TicketPriority.HIGH,
                expected_team=Team.PRIVACY,
                requires_compliance=True,
                must_resolve=False,
            ),
        ],
    ),
}


def list_task_ids() -> List[str]:
    return list(TASKS.keys())
