"""Deterministic event injection for the Compliance Audit environment.

This module injects mid-episode incidents (policy updates, system outages, record amendments)
based on a deterministic schedule computed from (task_id, seed).

The full event schedule is always returned in info["event_schedule"] at
reset() so graders remain fully deterministic regardless of agent actions.

Event Types
-----------
POLICY_UPDATE      — A rule's threshold changes at step N.
                     e.g. overtime threshold drops from 48 → 40 h.
SYSTEM_OUTAGE      — A record becomes temporarily inaccessible for D steps.
                     Attempting to inspect/audit it returns system_unavailable.
RECORD_AMENDMENT   — A field value is corrected at step N (simulates a data
                     correction ticket being processed mid-audit).

Determinism Guarantee
---------------------
For any (task_id, seed) pair the event schedule is byte-for-byte identical
across all runs.  The schedule is computed once at reset() and stored in
AuditState.event_schedule.  The environment applies events in tick() at
each step.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .models import EventEntry, EventType


# ---------------------------------------------------------------------------
# Per-task event templates
# Each entry defines a CANDIDATE event.  The EventScheduler picks a
# deterministic subset based on the seed and assigns trigger_steps.
# ---------------------------------------------------------------------------

_TASK_EVENT_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "easy_basic_audit": [
        # Simple outage on a record to test agent resilience (fires at step 5)
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "E003",
            "payload": {"duration_steps": 3},
            "description": (
                "SYSTEM OUTAGE: Record E003 is temporarily inaccessible due to "
                "a HR system maintenance window. Retry in 3 steps."
            ),
        },
    ],
    "medium_mixed_audit": [
        # Overtime policy tightened mid-audit (no R7 in this task, so cosmetic —
        # but tests that agents read policy updates)
        {
            "event_type": EventType.POLICY_UPDATE,
            "rule_id": "R1",
            "payload": {
                "field": "minor_hours_threshold",
                "old_value": 8,
                "new_value": 6,
                "note": "New minor-employment directive: threshold reduced to 6 h/week.",
            },
            "description": (
                "POLICY UPDATE: Minor employee hours threshold changed from 8 to 6 h/week. "
                "Re-evaluate any records with minor employees working 7-8 hours."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "M007",
            "payload": {"duration_steps": 4},
            "description": (
                "SYSTEM OUTAGE: Record M007 is locked pending HR investigation. "
                "Inaccessible for 4 steps."
            ),
        },
    ],
    "hard_complex_audit": [
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "H012",
            "payload": {
                "field": "contract_end",
                "old_value": "2023-06-01",
                "new_value": "2024-06-01",
                "note": "Contract was renewed retroactively. R5 no longer applies.",
            },
            "description": (
                "RECORD AMENDMENT: H012 contract_end corrected from 2023-06-01 to "
                "2024-06-01 after payroll verified renewal paperwork."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "H019",
            "payload": {"duration_steps": 5},
            "description": (
                "SYSTEM OUTAGE: Record H019 under legal hold — inaccessible for 5 steps."
            ),
        },
    ],
    "finance_sox_audit": [
        {
            "event_type": EventType.POLICY_UPDATE,
            "rule_id": "R7",
            "payload": {
                "field": "overtime_hours_threshold",
                "old_value": 48,
                "new_value": 44,
                "note": (
                    "Finance department overtime policy tightened: >44 h now requires approval. "
                    "Re-check all records with hours between 45-48."
                ),
            },
            "description": (
                "POLICY UPDATE: Finance overtime approval threshold lowered from 48 to 44 h/week. "
                "Re-evaluate records F004, F008, and any others with hours 45-47."
            ),
        },
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "F002",
            "payload": {
                "field": "background_check",
                "old_value": False,
                "new_value": True,
                "note": "Background check completed and processed by HR. R6 violation resolved.",
            },
            "description": (
                "RECORD AMENDMENT: F002 (Priya Sharma) background check updated to True. "
                "R6 violation is now resolved — do NOT flag R6 on F002 after this step."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "F010",
            "payload": {"duration_steps": 6},
            "description": (
                "SYSTEM OUTAGE: F010 (Yuki Tanaka) record locked by Legal for 6 steps. "
                "Inspect after the outage window."
            ),
        },
    ],
    "gdpr_privacy_audit": [
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "G005",
            "payload": {
                "field": "gdpr_consent",
                "old_value": False,
                "new_value": True,
                "note": "GDPR consent form received and filed. R9 no longer applies to G005.",
            },
            "description": (
                "RECORD AMENDMENT: G005 (Elise Bouchard) GDPR consent updated to True. "
                "R9 is now resolved — do NOT flag G005:R9 after this step."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "G008",
            "payload": {"duration_steps": 3},
            "description": (
                "SYSTEM OUTAGE: G008 (Hector Vega) account archived — inaccessible for 3 steps."
            ),
        },
    ],
    "data_integrity_audit": [
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "DI003",
            "payload": {
                "field": "salary",
                "old_value": None,
                "new_value": 55000,
                "note": "Missing salary field backfilled from payroll system.",
            },
            "description": (
                "RECORD AMENDMENT: DI003 salary field populated (55000). "
                "R10 is now resolved for DI003 — but flag it if inspected before this step."
            ),
        },
    ],
    "regulatory_storm_audit": [
        # Multiple simultaneous events to stress-test agents
        {
            "event_type": EventType.POLICY_UPDATE,
            "rule_id": "R7",
            "payload": {
                "field": "overtime_hours_threshold",
                "old_value": 48,
                "new_value": 40,
                "note": (
                    "Emergency regulatory directive: overtime threshold reduced to 40 h. "
                    "ALL records with hours 41-48 now require overtime_approved."
                ),
            },
            "description": (
                "POLICY UPDATE [CRITICAL]: Overtime threshold reduced 48→40 h/week. "
                "Re-evaluate ALL records with 41-47 hours."
            ),
        },
        {
            "event_type": EventType.POLICY_UPDATE,
            "rule_id": "R1",
            "payload": {
                "field": "minor_hours_threshold",
                "old_value": 8,
                "new_value": 4,
                "note": "New child labour directive: threshold halved.",
            },
            "description": (
                "POLICY UPDATE: Minor hours threshold reduced from 8 to 4 h/week. "
                "Re-inspect records with minor employees (age<18)."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "RS007",
            "payload": {"duration_steps": 8},
            "description": (
                "SYSTEM OUTAGE: RS007 under regulatory hold for 8 steps."
            ),
        },
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "RS015",
            "payload": {"duration_steps": 6},
            "description": (
                "SYSTEM OUTAGE: RS015 locked by legal compliance team for 6 steps."
            ),
        },
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "RS003",
            "payload": {
                "field": "compliance_training",
                "old_value": False,
                "new_value": True,
                "note": "Training completion backfilled by LMS system.",
            },
            "description": (
                "RECORD AMENDMENT: RS003 compliance_training corrected to True. "
                "R8 is resolved for RS003 after this step."
            ),
        },
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "RS019",
            "payload": {
                "field": "background_check",
                "old_value": None,
                "new_value": True,
                "note": "Background check verified and logged.",
            },
            "description": (
                "RECORD AMENDMENT: RS019 background_check set to True. "
                "R6 resolved for RS019 after this step."
            ),
        },
    ],
    "streaming_long_horizon": [
        # Step ~50: R8 resolved — training record backfilled for Iris Ferreira early in the sweep
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "S008",
            "trigger_step": 50,
            "payload": {
                "field": "compliance_training",
                "old_value": False,
                "new_value": True,
                "note": "LMS backfill: training completion logged retroactively.",
            },
            "description": (
                "RECORD AMENDMENT: S008 (Iris Ferreira) compliance_training updated to True. "
                "R8 violation is now resolved — do NOT flag R8 on S008 after this step."
            ),
        },
        # Step ~120: R9 resolved — GDPR consent form received for Henry Bassett
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "S120",
            "trigger_step": 120,
            "payload": {
                "field": "gdpr_consent",
                "old_value": False,
                "new_value": True,
                "note": "GDPR consent form signed and filed by data subject.",
            },
            "description": (
                "RECORD AMENDMENT: S120 gdpr_consent updated to True. "
                "R9 violation is now resolved — do NOT flag R9 on S120 after this step."
            ),
        },
        # Step ~230: SYSTEM_OUTAGE — legal hold on a record with multiple violations
        {
            "event_type": EventType.SYSTEM_OUTAGE,
            "record_id": "S224",
            "trigger_step": 230,
            "payload": {"duration_steps": 20},
            "description": (
                "SYSTEM OUTAGE: S224 (Omar Diaz) placed under legal hold — "
                "inaccessible for 20 steps. Continue auditing other records and retry after."
            ),
        },
        # Step ~310: R5 resolved — contract renewal processed for Ada Lewis
        {
            "event_type": EventType.RECORD_AMENDMENT,
            "record_id": "S280",
            "trigger_step": 310,
            "payload": {
                "field": "contract_end",
                "old_value": "2023-06-01",
                "new_value": "2026-12-31",
                "note": "Contract renewal backdated and approved by HR director.",
            },
            "description": (
                "RECORD AMENDMENT: S280 (Ada Lewis) contract_end updated to 2026-12-31. "
                "R5 violation is now resolved — do NOT flag R5 on S280 after this step."
            ),
        },
    ],
}

# Default trigger steps (index in template list → step number)
# These are adjusted by the seed for genuine variation
_BASE_TRIGGER_STEPS = [5, 12, 20, 30, 40, 50]


def _deterministic_offset(task_id: str, seed: int, idx: int) -> int:
    """Compute a deterministic small offset in [-3, +3] using a hash."""
    h = hashlib.sha256(f"{task_id}:{seed}:{idx}".encode()).digest()
    # Use first byte mod 7, then shift to range [-3, +3]
    return (h[0] % 7) - 3


class EventScheduler:
    """Compute and apply deterministic mid-episode events.

    Usage
    -----
    scheduler = EventScheduler(task_id, seed)
    schedule = scheduler.build_schedule()   # call once at reset()
    fired = scheduler.tick(step, state)     # call once per step()
    """

    def __init__(self, task_id: str, seed: int = 42) -> None:
        self.task_id = task_id
        self.seed = seed

    def build_schedule(self) -> List[EventEntry]:
        """Return the deterministic event list for this (task_id, seed).

        If a template entry contains a "trigger_step" key it is used as the
        base step directly (seed offset still applied on top).  This allows
        long-horizon tasks like streaming_long_horizon to spread events across
        hundreds of steps rather than being crowded into _BASE_TRIGGER_STEPS.
        """
        templates = _TASK_EVENT_TEMPLATES.get(self.task_id, [])
        events: List[EventEntry] = []
        for idx, tmpl in enumerate(templates):
            if "trigger_step" in tmpl:
                # Per-template override: use the explicit step, still jitter by seed
                base_step = tmpl["trigger_step"]
            else:
                base_step = _BASE_TRIGGER_STEPS[idx] if idx < len(_BASE_TRIGGER_STEPS) else 15
            offset = _deterministic_offset(self.task_id, self.seed, idx)
            trigger_step = max(1, base_step + offset)
            events.append(
                EventEntry(
                    event_type=tmpl["event_type"],
                    trigger_step=trigger_step,
                    record_id=tmpl.get("record_id"),
                    rule_id=tmpl.get("rule_id"),
                    payload=dict(tmpl.get("payload", {})),
                    description=tmpl.get("description", ""),
                    fired=False,
                )
            )
        return events

    @staticmethod
    def apply_events(
        step: int,
        event_schedule: List[EventEntry],
        records: Dict[str, Any],
        policy_overrides: Dict[str, Any],
    ) -> List[EventEntry]:
        """Apply any events whose trigger_step == current step.

        Modifies records and policy_overrides in-place.
        Returns the list of events that fired this step.
        """
        fired_this_step: List[EventEntry] = []
        for event in event_schedule:
            if event.fired or event.trigger_step != step:
                continue

            if event.event_type == EventType.POLICY_UPDATE:
                rule_id = event.rule_id
                if rule_id:
                    if rule_id not in policy_overrides:
                        policy_overrides[rule_id] = {}
                    field = event.payload.get("field")
                    new_value = event.payload.get("new_value")
                    if field and new_value is not None:
                        policy_overrides[rule_id][field] = new_value

            elif event.event_type == EventType.SYSTEM_OUTAGE:
                rec_id = event.record_id
                duration = event.payload.get("duration_steps", 3)
                if rec_id and rec_id in records:
                    records[rec_id].system_outage = True
                    records[rec_id].outage_ends_at_step = step + duration

            elif event.event_type == EventType.RECORD_AMENDMENT:
                rec_id = event.record_id
                field = event.payload.get("field")
                new_value = event.payload.get("new_value")
                if rec_id and field and rec_id in records:
                    # Save original if not already saved
                    if not records[rec_id].original_fields:
                        records[rec_id].original_fields = dict(records[rec_id].fields)
                    records[rec_id].fields[field] = new_value

            event.fired = True
            fired_this_step.append(event)

        # Expire any active outages
        for rec in records.values():
            if rec.system_outage and step >= rec.outage_ends_at_step:
                rec.system_outage = False

        return fired_this_step
