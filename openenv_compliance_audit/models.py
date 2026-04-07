from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    INSPECT_RECORD = "inspect_record"
    APPLY_RULE = "apply_rule"
    FLAG_VIOLATION = "flag_violation"
    MARK_COMPLIANT = "mark_compliant"
    GENERATE_REPORT = "generate_report"
    FINISH = "finish"


class AuditAction(BaseModel):
    """Action that the agent submits to the environment each step."""

    action_type: ActionType = Field(description="The action to perform")
    record_id: Optional[str] = Field(default=None, description="Target record ID")
    rule_id: Optional[str] = Field(default=None, description="Rule to apply or flag")

    @field_validator("record_id", "rule_id")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v is not None else v


class RecordView(BaseModel):
    """Agent-visible projection of a record."""

    record_id: str
    fields: Dict[str, Any]
    inspected: bool = False
    marked_compliant: bool = False
    flags: List[str] = Field(default_factory=list, description="rule_ids flagged as violations")


class ViolationEntry(BaseModel):
    record_id: str
    rule_id: str
    description: str


class AuditObservation(BaseModel):
    """Observation returned on every step."""

    task_id: str
    task_title: str
    objective: str
    available_rules: List[Dict[str, str]]
    step_index: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    remaining_steps: int = Field(ge=0)
    visible_records: List[RecordView]
    checked_records: List[str] = Field(description="record_ids that have been inspected")
    violations_found: List[ViolationEntry]
    action_history: List[str]
    last_action_error: Optional[str] = None


class AuditReward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    rationale: str


class RecordInternalState(BaseModel):
    """Ground-truth full state for one record."""

    record_id: str
    fields: Dict[str, Any]
    # Expected violations: set of (rule_id,) tuples
    expected_violations: List[str] = Field(
        default_factory=list, description="rule_ids that should be flagged"
    )
    inspected: bool = False
    marked_compliant: bool = False
    flagged_violations: List[str] = Field(default_factory=list)
    rules_applied: List[str] = Field(default_factory=list)


class AuditState(BaseModel):
    benchmark: str = "openenv_compliance_audit"
    task_id: str
    task_title: str
    objective: str
    difficulty: Literal["easy", "medium", "hard"]
    active_rule_ids: List[str]
    step_count: int = 0
    max_steps: int
    done: bool = False
    last_action_error: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)
    penalties: float = 0.0
    records: Dict[str, RecordInternalState] = Field(default_factory=dict)
    report_generated: bool = False


class StepResult(BaseModel):
    observation: AuditObservation
    reward: float = Field(ge=-1.0, le=1.0)
    done: bool
    info: Dict[str, object]
