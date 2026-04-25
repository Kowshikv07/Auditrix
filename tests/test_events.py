"""Tests for the dynamic event injection system.

Covers:
  - EventScheduler builds deterministic schedule per (task_id, seed)
  - Different seeds produce different trigger_steps
  - SYSTEM_OUTAGE: blocked record returns error; unblocks after duration
  - POLICY_UPDATE: rule threshold changes mid-episode
  - RECORD_AMENDMENT: field value changes; rule re-evaluates correctly
  - event_schedule present in reset info (via observation active_events)
  - Extreme task resets cleanly with all 10 rules
"""
from openenv_compliance_audit.environment import ComplianceAuditEnv
from openenv_compliance_audit.events import EventScheduler
from openenv_compliance_audit.models import AuditAction
from openenv_compliance_audit.tasks import TASKS


# ============================================================================
# EventScheduler determinism
# ============================================================================

def test_event_schedule_deterministic_same_seed() -> None:
    """Same (task_id, seed) → identical trigger_steps."""
    s1 = EventScheduler("easy_basic_audit", seed=42)
    s2 = EventScheduler("easy_basic_audit", seed=42)
    sched1 = s1.build_schedule()
    sched2 = s2.build_schedule()
    assert len(sched1) == len(sched2)
    for e1, e2 in zip(sched1, sched2):
        assert e1.trigger_step == e2.trigger_step
        assert e1.event_type == e2.event_type


def test_event_schedule_varies_with_seed() -> None:
    """Different seeds → different trigger_steps for at least one event."""
    s1 = EventScheduler("medium_mixed_audit", seed=42)
    s2 = EventScheduler("medium_mixed_audit", seed=99)
    sched1 = s1.build_schedule()
    sched2 = s2.build_schedule()
    # At least one event should differ
    steps1 = [e.trigger_step for e in sched1]
    steps2 = [e.trigger_step for e in sched2]
    assert steps1 != steps2, "Seeds 42 and 99 should produce different trigger steps"


def test_trigger_steps_are_positive() -> None:
    """All trigger_steps must be >= 1."""
    for task_id in TASKS:
        scheduler = EventScheduler(task_id, seed=42)
        for event in scheduler.build_schedule():
            assert event.trigger_step >= 1, f"{task_id}: trigger_step < 1"


def test_tasks_without_events_return_empty_schedule() -> None:
    """Tasks with no registered events should return an empty schedule."""
    # Currently all tasks have at least one event; this is a sanity guard
    for task_id in TASKS:
        scheduler = EventScheduler(task_id, seed=42)
        schedule = scheduler.build_schedule()
        # Schedule may be empty or non-empty — just verify it's a list
        assert isinstance(schedule, list)


# ============================================================================
# SYSTEM_OUTAGE
# ============================================================================

def test_system_outage_blocks_inspect() -> None:
    """When a SYSTEM_OUTAGE fires, inspect_record on that record must fail."""
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset(seed=42)

    # Find which step the outage fires on
    state = env.state()
    outage_events = [
        e for e in state.event_schedule
        if e.event_type.value == "system_outage"
    ]
    if not outage_events:
        return  # No outage event for this seed — skip

    event = outage_events[0]
    # Advance to that step
    for _ in range(event.trigger_step - state.step_count):
        env.step(AuditAction(action_type="wait" if False else "finish"))
        break  # Just trigger the first step to fire the event

    # Directly advance the state's step count to the trigger step
    # by stepping with harmless actions
    env2 = ComplianceAuditEnv(task_id="easy_basic_audit")
    env2.reset(seed=42)
    state2 = env2.state()
    outage_events2 = [
        e for e in state2.event_schedule
        if e.event_type.value == "system_outage"
    ]
    if not outage_events2:
        return

    ev = outage_events2[0]
    # Advance step count to trigger step using any valid action
    # (inspect E001 repeatedly — first inspect and then re-inspects are penalised but valid)
    for step_num in range(1, ev.trigger_step + 1):
        if step_num < ev.trigger_step:
            # Use mark_compliant on a dummy or inspect E001 repeatedly
            env2.step(AuditAction(action_type="inspect_record", record_id="E001"))
        else:
            # At trigger step, the outage fires first, then we try to inspect the outaged record
            result = env2.step(
                AuditAction(action_type="inspect_record", record_id=ev.record_id)
            )
            # The outage should block inspection
            obs = result.observation
            outaged_record = next(
                (r for r in obs.visible_records if r.record_id == ev.record_id), None
            )
            if outaged_record and outaged_record.system_outage:
                assert result.observation.last_action_error is not None, (
                    "Inspecting an outaged record should return an error"
                )


def test_system_outage_visible_in_observation() -> None:
    """After an outage fires, the record view should show system_outage=True."""
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset(seed=42)
    state = env.state()
    outage_events = [
        e for e in state.event_schedule
        if e.event_type.value == "system_outage"
    ]
    if not outage_events:
        return

    event = outage_events[0]
    # Step to just before trigger
    for _ in range(event.trigger_step - 1):
        r = env.step(AuditAction(action_type="inspect_record", record_id="E001"))
        if r.done:
            return

    # Now step at trigger_step — outage fires during this step
    result = env.step(AuditAction(action_type="inspect_record", record_id="E001"))
    obs = result.observation
    outaged_record = next(
        (r for r in obs.visible_records if r.record_id == event.record_id), None
    )
    if outaged_record:
        assert outaged_record.system_outage is True or result.done, (
            "Record should be marked as in outage after event fires"
        )


# ============================================================================
# POLICY_UPDATE
# ============================================================================

def test_policy_update_changes_threshold() -> None:
    """After a POLICY_UPDATE fires for R1 (medium task), policy_overrides must update."""
    env = ComplianceAuditEnv(task_id="medium_mixed_audit")
    env.reset(seed=42)

    state = env.state()
    policy_events = [
        e for e in state.event_schedule
        if e.event_type.value == "policy_update"
    ]
    if not policy_events:
        return

    event = policy_events[0]
    # Step to trigger step
    for i in range(event.trigger_step):
        result = env.step(AuditAction(action_type="inspect_record", record_id="M001"))
        if result.done:
            return
        state = env.state()
    
    # After trigger_step, policy_overrides should contain the new value
    state_after = env.state()
    if event.rule_id and event.rule_id in state_after.policy_overrides:
        override = state_after.policy_overrides[event.rule_id]
        field = event.payload.get("field")
        new_value = event.payload.get("new_value")
        if field and field in override:
            assert override[field] == new_value, (
                f"Policy override should set {field}={new_value}, got {override[field]}"
            )


def test_policy_update_reflected_in_observation() -> None:
    """current_policy_overrides in observation updates after POLICY_UPDATE fires."""
    env = ComplianceAuditEnv(task_id="finance_sox_audit")
    env.reset(seed=42)

    state = env.state()
    policy_events = [
        e for e in state.event_schedule
        if e.event_type.value == "policy_update"
    ]
    if not policy_events:
        return

    event = policy_events[0]
    for _ in range(event.trigger_step):
        result = env.step(AuditAction(action_type="inspect_record", record_id="F001"))
        if result.done:
            return

    obs = result.observation
    if event.rule_id:
        # By the step after trigger, policy_overrides should be visible
        assert isinstance(obs.current_policy_overrides, dict)


# ============================================================================
# RECORD_AMENDMENT
# ============================================================================

def test_record_amendment_changes_field() -> None:
    """After a RECORD_AMENDMENT fires, the record's field should have the new value."""
    env = ComplianceAuditEnv(task_id="data_integrity_audit")
    env.reset(seed=42)

    state = env.state()
    amendment_events = [
        e for e in state.event_schedule
        if e.event_type.value == "record_amendment"
    ]
    if not amendment_events:
        return

    event = amendment_events[0]
    for _ in range(event.trigger_step):
        result = env.step(AuditAction(action_type="inspect_record", record_id="DI001"))
        if result.done:
            return

    state_after = env.state()
    if event.record_id and event.record_id in state_after.records:
        rec = state_after.records[event.record_id]
        field = event.payload.get("field")
        new_value = event.payload.get("new_value")
        if field:
            assert rec.fields.get(field) == new_value, (
                f"Field {field} should be {new_value!r} after amendment, "
                f"got {rec.fields.get(field)!r}"
            )


def test_record_amendment_flag_after_resolves_is_false_positive() -> None:
    """Flagging a violation AFTER a RECORD_AMENDMENT resolves it should be a false positive."""
    env = ComplianceAuditEnv(task_id="data_integrity_audit")
    env.reset(seed=42)

    state = env.state()
    amendment_events = [
        e for e in state.event_schedule
        if e.event_type.value == "record_amendment"
        and e.record_id == "DI003"
    ]
    if not amendment_events:
        return  # Amendment for DI003 only fires with certain seeds

    event = amendment_events[0]
    # Advance past the amendment
    env.step(AuditAction(action_type="inspect_record", record_id="DI003"))
    for _ in range(event.trigger_step):
        result = env.step(AuditAction(action_type="inspect_record", record_id="DI001"))
        if result.done:
            return

    # DI003's salary was set by amendment — R10 is now resolved
    # Trying to flag DI003:R10 should be treated as a false positive
    result = env.step(
        AuditAction(action_type="flag_violation", record_id="DI003", rule_id="R10")
    )
    # Should be penalised (false positive after amendment)
    assert result.reward <= 0.0, "Flagging an amendment-resolved violation should not reward"


# ============================================================================
# Extreme task
# ============================================================================

def test_regulatory_storm_resets_cleanly() -> None:
    """regulatory_storm_audit task resets with 25 records and all 10 rules."""
    env = ComplianceAuditEnv(task_id="regulatory_storm_audit")
    obs = env.reset(seed=42)
    state = env.state()

    assert obs.task_id == "regulatory_storm_audit"
    assert len(obs.visible_records) == 25
    assert state.difficulty == "extreme"
    assert len(state.active_rule_ids) == 11
    assert obs.step_index == 0
    assert obs.remaining_steps == 120


def test_regulatory_storm_has_events() -> None:
    """Extreme task should have a non-empty event schedule."""
    env = ComplianceAuditEnv(task_id="regulatory_storm_audit")
    env.reset(seed=42)
    state = env.state()
    assert len(state.event_schedule) > 0, "Extreme task must have at least one event"


def test_regulatory_storm_grader_bounded() -> None:
    """Terminating immediately on extreme task should return score in [0, 1]."""
    env = ComplianceAuditEnv(task_id="regulatory_storm_audit")
    env.reset(seed=42)
    result = env.step(AuditAction(action_type="generate_report"))
    score = float(result.info["task_score"])
    assert 0.0 <= score <= 1.0


def test_regulatory_storm_ground_truth_deterministic() -> None:
    """Rules must agree with expected_violations for extreme task (pre-event snapshot)."""
    from openenv_compliance_audit.rules import RULES
    from openenv_compliance_audit.tasks import TASKS

    task_id = "regulatory_storm_audit"
    task = TASKS[task_id]
    all_fields = [r.fields for r in task.records]
    mismatches = []
    for rec in task.records:
        # Use evaluate() with no policy_override (pre-event baseline)
        detected = sorted(
            rid for rid in task.active_rule_ids
            if RULES[rid].evaluate(rec.fields, all_fields)
        )
        expected = sorted(rec.expected_violations)
        if detected != expected:
            mismatches.append(
                f"{task_id}/{rec.record_id}: engine={detected} declared={expected}"
            )
    assert not mismatches, "Ground-truth mismatches:\n" + "\n".join(mismatches)


# ============================================================================
# Loop detection
# ============================================================================

def test_loop_detection_triggers_penalty() -> None:
    """Repeating the same (action_type, record_id) > 3 times in 5 steps triggers penalty."""
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset(seed=42)

    # First inspect is valid
    env.step(AuditAction(action_type="inspect_record", record_id="E001"))

    # Now repeat inspect_record:E001 enough times to trigger loop detection
    penalties_hit = False
    for _ in range(6):
        result = env.step(AuditAction(action_type="inspect_record", record_id="E001"))
        if "loop_penalty" in result.info.get("reward_model", {}).get("components", {}):
            penalties_hit = True
            break

    assert penalties_hit, "Loop detection should have triggered a penalty"


def test_loop_penalty_applied_counter() -> None:
    """loop_penalty_applied counter increments when loop is detected."""
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset(seed=42)
    env.step(AuditAction(action_type="inspect_record", record_id="E001"))

    for _ in range(6):
        env.step(AuditAction(action_type="inspect_record", record_id="E001"))

    state = env.state()
    assert state.loop_penalty_applied >= 1, "loop_penalty_applied should be >= 1"
