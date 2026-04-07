from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    INSPECT_TICKET = "inspect_ticket"
    SET_PRIORITY = "set_priority"
    ASSIGN_TEAM = "assign_team"
    REQUEST_CUSTOMER_REPLY = "request_customer_reply"
    ADD_INTERNAL_NOTE = "add_internal_note"
    ESCALATE_COMPLIANCE = "escalate_compliance"
    RESOLVE_TICKET = "resolve_ticket"
    FINISH = "finish"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Team(str, Enum):
    BILLING = "billing"
    SUPPORT = "support"
    PRODUCT = "product"
    SECURITY = "security"
    FRAUD = "fraud"
    PRIVACY = "privacy"


class TicketTriageAction(BaseModel):
    action_type: ActionType = Field(description="The action to execute in the environment")
    ticket_id: Optional[str] = Field(default=None, description="Target ticket id")
    value: Optional[str] = Field(
        default=None,
        description="Optional payload. Used by set_priority, assign_team, add_internal_note, or resolve summary.",
    )

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return value.strip()


class TicketView(BaseModel):
    ticket_id: str
    subject: str
    customer_tier: Literal["free", "pro", "enterprise"]
    age_hours: int = Field(ge=0)
    inspected: bool
    priority: Optional[TicketPriority] = None
    assigned_team: Optional[Team] = None
    requested_customer_reply: bool = False
    compliance_escalated: bool = False
    resolved: bool = False


class TicketTriageObservation(BaseModel):
    task_id: str
    task_title: str
    objective: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    progress_score: float = Field(ge=0.0, le=1.0)
    visible_tickets: List[TicketView]
    action_history: List[str]
    last_action_error: Optional[str] = None


class TicketTriageReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float]
    rationale: str


class TicketInternalState(BaseModel):
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
    inspected: bool = False
    priority: Optional[TicketPriority] = None
    assigned_team: Optional[Team] = None
    requested_customer_reply: bool = False
    compliance_escalated: bool = False
    resolved: bool = False
    internal_notes: List[str] = Field(default_factory=list)


class TicketTriageState(BaseModel):
    benchmark: str = "openenv_ticket_triage"
    task_id: str
    task_title: str
    objective: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = 0
    max_steps: int
    done: bool = False
    last_action_error: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)
    penalties: float = 0.0
    tickets: Dict[str, TicketInternalState] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: TicketTriageObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, object]
