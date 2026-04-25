from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    INSPECT_RECORD = "inspect_record"
    APPLY_RULE = "apply_rule"
    REQUEST_EVIDENCE = "request_evidence"   # deliberate before flagging — free (0 reward, costs 1 step)
    FLAG_VIOLATION = "flag_violation"
    RETRACT_FLAG = "retract_flag"           # undo a flag; -0.10 (correct retract) / -0.20 (correct flag retracted)
    MARK_COMPLIANT = "mark_compliant"
    PRIORITIZE_RULES = "prioritize_rules"
    GENERATE_REPORT = "generate_report"
    FINISH = "finish"


class AuditAction(BaseModel):
    """Action that the agent submits to the environment each step."""

    action_type: ActionType = Field(description="The action to perform")
    record_id: Optional[str] = Field(default=None, description="Target record ID")
    rule_id: Optional[str] = Field(default=None, description="Rule to apply or flag")
    rule_priority_order: Optional[List[str]] = Field(
        default=None,
        description="Rule IDs in priority order for prioritize_rules action (enables multi-step planning)",
    )
    report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional structured audit report payload used with generate_report. "
            "Ignored by other actions."
        ),
    )

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
    system_outage: bool = Field(
        default=False,
        description="True when a SYSTEM_OUTAGE event makes this record temporarily inaccessible",
    )


class ViolationEntry(BaseModel):
    record_id: str
    rule_id: str
    description: str
    severity: str = Field(default="medium", description="critical | high | medium | low")


class DecisionTrace(BaseModel):
    """Structured explainability trace for apply_rule/request_evidence/flag_violation actions."""

    action_type: Literal["apply_rule", "request_evidence", "flag_violation"]
    record_id: str
    rule_id: str
    outcome: str
    verdict: str = Field(
        default="compliant",
        description="One of: violation | warning | insufficient_evidence | compliant",
    )
    confidence_tier: str = Field(
        default="high",
        description="One of: high | medium | low"
    )
    reason_codes: List[str] = Field(default_factory=list)
    rule_evidence: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Dynamic incident event models
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    POLICY_UPDATE = "policy_update"        # A rule's threshold changes mid-episode
    SYSTEM_OUTAGE = "system_outage"        # A record becomes temporarily inaccessible
    RECORD_AMENDMENT = "record_amendment"  # A field value is corrected mid-episode
    RULE_SUSPENSION = "rule_suspension"    # A rule is temporarily deactivated mid-episode


class EventEntry(BaseModel):
    """A deterministic incident event scheduled for a specific step.

    Events are injected based on (task_id, seed) — fully deterministic.
    The complete event_schedule is returned in info["event_schedule"] at reset()
    so graders remain deterministic regardless of when the agent fires an action.
    """

    event_type: EventType
    trigger_step: int = Field(ge=1, description="Step at which the event fires")
    record_id: Optional[str] = Field(default=None, description="Affected record (if applicable)")
    rule_id: Optional[str] = Field(default=None, description="Affected rule (if applicable)")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Event-specific data. "
            "POLICY_UPDATE: {field, old_value, new_value}. "
            "SYSTEM_OUTAGE: {duration_steps}. "
            "RECORD_AMENDMENT: {field, old_value, new_value}."
        ),
    )
    fired: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Audit confidence report section (submitted inside generate_report payload)
# ---------------------------------------------------------------------------

class AuditConfidenceReport(BaseModel):
    """Optional agent-submitted confidence section in the generate_report payload.

    Scored in report_quality_components under the key 'confidence_score'.
    Scoring rewards accurate evidence_coverage_ratio and penalises uncertain_flags
    that turned out to be correct violations (i.e. the agent was uncertain when it
    should have been confident).
    """

    evidence_coverage_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of records for which the agent has field-level evidence",
    )
    high_confidence_flags: List[str] = Field(
        default_factory=list,
        description="'record_id:rule_id' pairs agent is highly confident about",
    )
    uncertain_flags: List[str] = Field(
        default_factory=list,
        description="'record_id:rule_id' pairs where evidence was ambiguous",
    )
    reasoning: str = Field(default="", description="Free-text audit reasoning summary")


# ---------------------------------------------------------------------------
# Failure mode taxonomy (used in grader output and [END] log lines)
# ---------------------------------------------------------------------------

FailureMode = Literal[
    "false_positive",        # Flagged a compliant record
    "missed_violation",      # Missed a true violation
    "low_coverage",          # < 50% of records inspected
    "inefficiency",          # Used > 90% of max_steps
    "loop_exploit",          # Repeated low-information actions detected
    "report_inconsistency",  # Submitted report contradicts flagged state
    "none",                  # No failure mode detected
]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class AuditObservation(BaseModel):
    """Observation returned on every step (and reset)."""

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
    last_decision_trace: Optional[DecisionTrace] = None
    # Dynamic events fired so far
    active_events: List[EventEntry] = Field(
        default_factory=list,
        description="Events that have fired so far in this episode",
    )
    # Policy overrides currently in effect (rule_id → override_dict)
    current_policy_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Active rule-threshold overrides from POLICY_UPDATE events",
    )
    # Rule prioritization for multi-step planning in streaming tasks
    rule_priority_order: List[str] = Field(
        default_factory=list,
        description="Agent's strategic prioritization of rules to audit first",
    )
    # Loop exploit action signature (for agent observability)
    loop_exploit_signature: Optional[str] = Field(
        default=None,
        description="Action signature (action_type:record_id) that triggered the loop exploit detection",
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class AuditReward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    rationale: str
    failure_mode: FailureMode = "none"


# ---------------------------------------------------------------------------
# Internal state models
# ---------------------------------------------------------------------------

class RecordInternalState(BaseModel):
    """Ground-truth full internal state for one record."""

    record_id: str
    fields: Dict[str, Any]
    original_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-amendment field snapshot (populated on reset)",
    )
    expected_violations: List[str] = Field(
        default_factory=list, description="rule_ids that should be flagged"
    )
    inspected: bool = False
    system_outage: bool = False  # True while a SYSTEM_OUTAGE event is active for this record
    outage_ends_at_step: int = Field(
        default=0,
        description="Step at which the SYSTEM_OUTAGE for this record expires",
    )
    marked_compliant: bool = False
    flagged_violations: List[str] = Field(default_factory=list)
    rules_applied: List[str] = Field(default_factory=list)


class AuditState(BaseModel):
    benchmark: str = "openenv_compliance_audit"
    task_id: str
    task_title: str
    objective: str
    difficulty: Literal["easy", "medium", "hard", "extreme", "streaming"]
    active_rule_ids: List[str]
    step_count: int = 0
    max_steps: int
    done: bool = False
    last_action_error: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)
    penalties: float = 0.0
    records: Dict[str, RecordInternalState] = Field(default_factory=dict)
    report_generated: bool = False
    last_decision_trace: Optional[DecisionTrace] = None
    # Dynamic event state
    event_schedule: List[EventEntry] = Field(
        default_factory=list,
        description="Full deterministic event schedule for this episode",
    )
    policy_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Currently active rule-threshold overrides keyed by rule_id",
    )
    # Anti-exploit: rolling action window for loop detection
    recent_action_window: List[str] = Field(
        default_factory=list,
        description="Last 5 (action_type:record_id) signatures for loop detection",
    )
    loop_penalty_applied: int = Field(
        default=0,
        description="Number of loop penalties applied this episode",
    )
    # Multi-step planning state
    rule_priority_order: List[str] = Field(
        default_factory=list,
        description="Agent-specified rule priority order for strategic audit planning",
    )
    rule_priority_set: bool = Field(
        default=False, description="Whether agent has called prioritize_rules"
    )
    reward_history: List[float] = Field(
        default_factory=list, description="All rewards given this episode"
    )
    loop_exploit_signature: Optional[str] = Field(
        default=None,
        description="Action signature (action_type:record_id) that triggered loop exploit detection",
    )
    # Warning call tracking: list of (record_id, rule_id) that returned 'warning' verdict
    warning_calls: List[str] = Field(
        default_factory=list,
        description="'record_id:rule_id' pairs where apply_rule returned verdict=warning",
    )
    # Suspended rule tracking
    suspended_rule_ids: List[str] = Field(
        default_factory=list,
        description="rule_ids temporarily deactivated by RULE_SUSPENSION events",
    )


class StepResult(BaseModel):
    observation: AuditObservation
    reward: float = Field(ge=-1.0, le=1.0)
    done: bool
    info: Dict[str, object]
