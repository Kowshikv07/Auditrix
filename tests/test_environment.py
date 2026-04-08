"""Tests for the compliance audit environment.

Covers:
  - reset() consistency
  - Perfect score on easy task
  - Inspection guard
  - False positive penalty
  - Repeat action penalty
  - Max-step termination
  - All 5 tasks reset cleanly
  - Grader score bounded [0, 1]
  - Rule ground truth for all tasks (deterministic evaluation)
  - finance_sox_audit: new rules R6, R7, R8
  - gdpr_privacy_audit: new rules R5, R8, R9 with exemption logic
  - Edge cases for new rules
"""
from openenv_compliance_audit.environment import ComplianceAuditEnv
from openenv_compliance_audit.models import AuditAction
from openenv_compliance_audit.rules import RULES
from openenv_compliance_audit.tasks import TASKS


# ===========================================================================
# Helpers
# ===========================================================================

def _run_perfect_audit(task_id: str) -> float:
    """Run the ground-truth optimal trajectory and return final score.

    Uses minimal steps (no apply_rule) so the trajectory never exceeds max_steps:
      easy  : 5+2+3+1  = 11  (max 25)
      medium: 12+9+3+1 = 25  (max 50)
      hard  : 20+15+6+1= 42  (max 100)
      sox   : 15+17+3+1= 36  (max 80)
      gdpr  : 10+9+1+1 = 21  (max 50)
    """
    env = ComplianceAuditEnv(task_id=task_id)
    env.reset()
    task = TASKS[task_id]

    # 1. Inspect all records
    for rec in task.records:
        env.step(AuditAction(action_type="inspect_record", record_id=rec.record_id))

    # 2. Flag every true violation (ground truth only — no false positives)
    for rec in task.records:
        for rule_id in rec.expected_violations:
            env.step(AuditAction(action_type="flag_violation",
                                 record_id=rec.record_id, rule_id=rule_id))

    # 3. Mark truly compliant records
    for rec in task.records:
        if not rec.expected_violations:
            env.step(AuditAction(action_type="mark_compliant",
                                 record_id=rec.record_id))

    # 4. Generate report — task_score is always present in info (even if already done)
    result = env.step(AuditAction(action_type="generate_report"))
    return float(result.info["task_score"])


# ===========================================================================
# reset() consistency
# ===========================================================================

def test_reset_returns_clean_state() -> None:
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    obs = env.reset()
    state = env.state()

    assert obs.task_id == "easy_basic_audit"
    assert state.task_id == "easy_basic_audit"
    assert state.step_count == 0
    assert state.done is False
    assert len(obs.visible_records) == 5
    assert obs.checked_records == []
    assert obs.violations_found == []
    assert obs.remaining_steps == obs.max_steps


# ===========================================================================
# Perfect scores on all 5 tasks
# ===========================================================================

def test_easy_task_perfect_score() -> None:
    """Optimal trajectory on easy_basic_audit must score >= 0.90."""
    score = _run_perfect_audit("easy_basic_audit")
    assert score >= 0.90, f"Expected >= 0.90, got {score:.4f}"


def test_medium_task_perfect_score() -> None:
    """Optimal trajectory on medium_mixed_audit must score >= 0.85."""
    score = _run_perfect_audit("medium_mixed_audit")
    assert score >= 0.85, f"Expected >= 0.85, got {score:.4f}"


def test_hard_task_perfect_score() -> None:
    """Optimal trajectory on hard_complex_audit must score >= 0.80."""
    score = _run_perfect_audit("hard_complex_audit")
    assert score >= 0.80, f"Expected >= 0.80, got {score:.4f}"


def test_finance_sox_perfect_score() -> None:
    """Optimal trajectory on finance_sox_audit must score >= 0.80."""
    score = _run_perfect_audit("finance_sox_audit")
    assert score >= 0.80, f"Expected >= 0.80, got {score:.4f}"


def test_data_integrity_perfect_score() -> None:
    """Optimal trajectory on data_integrity_audit must score >= 0.85."""
    score = _run_perfect_audit("data_integrity_audit")
    assert score >= 0.85, f"Expected >= 0.85, got {score:.4f}"


def test_gdpr_privacy_perfect_score() -> None:
    """Optimal trajectory on gdpr_privacy_audit must score >= 0.85."""
    score = _run_perfect_audit("gdpr_privacy_audit")
    assert score >= 0.85, f"Expected >= 0.85, got {score:.4f}"


# ===========================================================================
# Guard: record must be inspected before audit actions
# ===========================================================================

def test_audit_requires_inspection_first() -> None:
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset()
    result = env.step(AuditAction(action_type="apply_rule", record_id="E001", rule_id="R1"))
    assert result.observation.last_action_error is not None
    assert result.reward == 0.0


# ===========================================================================
# Penalty: false positive flag
# ===========================================================================

def test_false_positive_flag_penalised() -> None:
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="E003"))
    result = env.step(AuditAction(action_type="flag_violation", record_id="E003", rule_id="R1"))
    assert result.reward < 0.0


# ===========================================================================
# Penalty: repeat action
# ===========================================================================

def test_repeat_inspect_is_penalised() -> None:
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="E001"))
    result = env.step(AuditAction(action_type="inspect_record", record_id="E001"))
    assert result.reward < 0.0


# ===========================================================================
# Termination: max steps
# ===========================================================================

def test_episode_ends_at_max_steps() -> None:
    env = ComplianceAuditEnv(task_id="easy_basic_audit")
    env.reset()
    max_steps = env.state().max_steps

    result = None
    for _ in range(max_steps + 5):
        if result and result.done:
            break
        result = env.step(AuditAction(action_type="inspect_record", record_id="E001"))

    assert result is not None
    assert result.done is True


# ===========================================================================
# All 5 tasks reset cleanly
# ===========================================================================

def test_all_tasks_reset_cleanly() -> None:
    for tid in TASKS:
        env = ComplianceAuditEnv(task_id=tid)
        obs = env.reset()
        assert obs.task_id == tid
        assert obs.step_index == 0
        assert len(obs.visible_records) > 0
        assert obs.violations_found == []


# ===========================================================================
# Grader score bounded [0, 1]
# ===========================================================================

def test_grader_score_bounded() -> None:
    for tid in TASKS:
        env = ComplianceAuditEnv(task_id=tid)
        env.reset()
        result = env.step(AuditAction(action_type="generate_report"))
        score = float(result.info["task_score"])
        assert 0.0 <= score <= 1.0, f"Score out of bounds for {tid}: {score}"


# ===========================================================================
# Deterministic ground truth: RULES match expected_violations in all tasks
# ===========================================================================

def test_all_task_ground_truths_deterministic() -> None:
    """Rules must agree with expected_violations for every record in every task."""
    mismatches = []
    for task_id, task in TASKS.items():
        all_fields = [r.fields for r in task.records]
        for rec in task.records:
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


# ===========================================================================
# New rule unit tests — R6: Background check
# ===========================================================================

def test_r6_sensitive_role_without_background_check() -> None:
    """manager/director/finance_manager/accountant without background_check=True → violation."""
    sensitive_records = [
        {"role": "manager",         "background_check": False},
        {"role": "director",        "background_check": None},
        {"role": "finance_manager", "background_check": False},
        {"role": "accountant"},                                  # field missing
        {"role": "cfo",             "background_check": False},
        {"role": "hr",              "background_check": False},
        {"role": "security",        "background_check": False},
    ]
    for fields in sensitive_records:
        assert RULES["R6"].evaluate(fields), f"R6 should trigger for {fields}"


def test_r6_no_violation_when_check_done_or_non_sensitive() -> None:
    assert not RULES["R6"].evaluate({"role": "manager",  "background_check": True})
    assert not RULES["R6"].evaluate({"role": "employee", "background_check": False})
    assert not RULES["R6"].evaluate({"role": "intern",   "background_check": False})
    assert not RULES["R6"].evaluate({"role": "analyst",  "background_check": False})
    assert not RULES["R6"].evaluate({"role": "contractor", "background_check": False})


# ===========================================================================
# New rule unit tests — R7: Unapproved overtime
# ===========================================================================

def test_r7_over_48h_without_approval() -> None:
    assert RULES["R7"].evaluate({"hours": 49, "overtime_approved": False})
    assert RULES["R7"].evaluate({"hours": 60})                       # field missing
    assert RULES["R7"].evaluate({"hours": 49, "overtime_approved": None})


def test_r7_edge_cases() -> None:
    # Exactly 48 hours is NOT a violation
    assert not RULES["R7"].evaluate({"hours": 48, "overtime_approved": False})
    # Approved overtime is fine
    assert not RULES["R7"].evaluate({"hours": 55, "overtime_approved": True})
    # Under 48 is always fine
    assert not RULES["R7"].evaluate({"hours": 47, "overtime_approved": False})


# ===========================================================================
# New rule unit tests — R8: Missing compliance training
# ===========================================================================

def test_r8_active_without_training() -> None:
    assert RULES["R8"].evaluate({"status": "active", "compliance_training": False})
    assert RULES["R8"].evaluate({"status": "active"})               # field missing
    assert RULES["R8"].evaluate({"compliance_training": False})     # no status → default active


def test_r8_inactive_employee_exempt() -> None:
    """Inactive employees must NOT be flagged under R8."""
    assert not RULES["R8"].evaluate({"status": "inactive", "compliance_training": False})
    assert not RULES["R8"].evaluate({"status": "inactive"})


def test_r8_training_done_compliant() -> None:
    assert not RULES["R8"].evaluate({"status": "active", "compliance_training": True})


# ===========================================================================
# New rule unit tests — R9: GDPR consent
# ===========================================================================

def test_r9_pii_access_without_consent() -> None:
    assert RULES["R9"].evaluate({"pii_access": True, "gdpr_consent": False})
    assert RULES["R9"].evaluate({"pii_access": True})               # consent field missing


def test_r9_no_pii_access_exempt() -> None:
    """Employees without PII access must NOT be flagged, regardless of consent field."""
    assert not RULES["R9"].evaluate({"pii_access": False, "gdpr_consent": False})
    assert not RULES["R9"].evaluate({"gdpr_consent": False})        # pii_access absent


def test_r9_compliant_with_consent() -> None:
    assert not RULES["R9"].evaluate({"pii_access": True, "gdpr_consent": True})


# ===========================================================================
# finance_sox_audit: specific scenario checks
# ===========================================================================

def test_finance_sox_f007_multi_violation() -> None:
    """F007 violates both R3 (salary too high) and R8 (no training)."""
    env = ComplianceAuditEnv(task_id="finance_sox_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="F007"))

    r3 = env.step(AuditAction(action_type="apply_rule", record_id="F007", rule_id="R3"))
    assert r3.reward > 0.0, "R3 should detect violation on F007"

    r8 = env.step(AuditAction(action_type="apply_rule", record_id="F007", rule_id="R8"))
    assert r8.reward > 0.0, "R8 should detect violation on F007"


def test_finance_sox_f015_hours48_not_overtime() -> None:
    """F015 has exactly 48 hours — R7 must NOT trigger (edge case)."""
    env = ComplianceAuditEnv(task_id="finance_sox_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="F015"))
    result = env.step(AuditAction(action_type="apply_rule", record_id="F015", rule_id="R7"))
    # R7 should NOT fire for 48 hours exactly
    assert result.reward == 0.0, "R7 must not trigger for exactly 48 hours"


def test_finance_sox_f008_fully_compliant() -> None:
    """F008 is fully compliant — flagging any rule is a false positive."""
    env = ComplianceAuditEnv(task_id="finance_sox_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="F008"))
    for rule_id in ["R3", "R6", "R7", "R8"]:
        result = env.step(AuditAction(action_type="flag_violation",
                                      record_id="F008", rule_id=rule_id))
        assert result.reward < 0.0, f"F008 should not have violation for {rule_id}"


# ===========================================================================
# gdpr_privacy_audit: exemption logic
# ===========================================================================

def test_gdpr_g008_inactive_exempt() -> None:
    """G008 is inactive — R5 and R8 must NOT trigger (expired contract + no training)."""
    env = ComplianceAuditEnv(task_id="gdpr_privacy_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="G008"))

    r5 = env.step(AuditAction(action_type="apply_rule", record_id="G008", rule_id="R5"))
    assert r5.reward == 0.0, "R5 should not trigger for inactive employee"

    r8 = env.step(AuditAction(action_type="apply_rule", record_id="G008", rule_id="R8"))
    assert r8.reward == 0.0, "R8 should not trigger for inactive employee"


def test_gdpr_g006_no_pii_access_exempt() -> None:
    """G006 has no PII access — R9 must NOT trigger even though gdpr_consent=False."""
    env = ComplianceAuditEnv(task_id="gdpr_privacy_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="G006"))
    result = env.step(AuditAction(action_type="apply_rule", record_id="G006", rule_id="R9"))
    assert result.reward == 0.0, "R9 should not trigger for record without pii_access"


# ===========================================================================
# New rule unit tests — R10: Missing required fields
# ===========================================================================

def test_r10_missing_salary() -> None:
    """Record without salary triggers R10."""
    assert RULES["R10"].evaluate({"id": 1, "name": "Alice", "role": "employee", "hours": 40})


def test_r10_missing_role() -> None:
    assert RULES["R10"].evaluate({"id": 1, "name": "Bob", "hours": 40, "salary": 50000})


def test_r10_missing_hours() -> None:
    assert RULES["R10"].evaluate({"id": 1, "name": "Carol", "role": "manager", "salary": 90000})


def test_r10_null_field_treated_as_missing() -> None:
    """Explicit None value counts as missing."""
    assert RULES["R10"].evaluate({"id": None, "name": "Dave", "role": "employee",
                                   "hours": 35, "salary": 50000})


def test_r10_complete_record_compliant() -> None:
    assert not RULES["R10"].evaluate({"id": 1, "name": "Eve", "role": "employee",
                                       "hours": 40, "salary": 60000})


def test_r10_zero_hours_not_missing() -> None:
    """hours=0 is a valid value (inactive staff) — must NOT trigger R10."""
    assert not RULES["R10"].evaluate({"id": 1, "name": "Frank", "role": "employee",
                                       "hours": 0, "salary": 30000})


def test_r10_missing_salary_does_not_trigger_r3() -> None:
    """R3 must skip gracefully when salary is missing — null-safe evaluation."""
    assert not RULES["R3"].evaluate({"id": 1, "name": "Grace", "role": "manager", "hours": 40})


# ===========================================================================
# data_integrity_audit: specific scenario checks
# ===========================================================================

def test_data_integrity_di003_r10_not_r3() -> None:
    """DI003 is missing salary: R10 fires but R3 must NOT (null-safe)."""
    env = ComplianceAuditEnv(task_id="data_integrity_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="DI003"))

    r10 = env.step(AuditAction(action_type="apply_rule", record_id="DI003", rule_id="R10"))
    assert r10.reward > 0.0, "R10 must detect missing salary on DI003"

    r3 = env.step(AuditAction(action_type="apply_rule", record_id="DI003", rule_id="R3"))
    assert r3.reward == 0.0, "R3 must skip gracefully when salary is absent"


def test_data_integrity_di004_di005_duplicate_id() -> None:
    """Both DI004 and DI005 share id=4 — both must trigger R4."""
    env = ComplianceAuditEnv(task_id="data_integrity_audit")
    env.reset()
    for rid in ["DI004", "DI005"]:
        env.step(AuditAction(action_type="inspect_record", record_id=rid))
    for rid in ["DI004", "DI005"]:
        result = env.step(AuditAction(action_type="apply_rule", record_id=rid, rule_id="R4"))
        assert result.reward > 0.0, f"R4 must detect duplicate ID on {rid}"


def test_data_integrity_di008_role_missing_r3_skips() -> None:
    """DI008 missing role: R10 fires; R3 skips (role not in salary range dict)."""
    env = ComplianceAuditEnv(task_id="data_integrity_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="DI008"))

    r10 = env.step(AuditAction(action_type="apply_rule", record_id="DI008", rule_id="R10"))
    assert r10.reward > 0.0, "R10 must fire for missing role"

    r3 = env.step(AuditAction(action_type="apply_rule", record_id="DI008", rule_id="R3"))
    assert r3.reward == 0.0, "R3 must skip when role is absent"


def test_gdpr_g005_double_violation() -> None:
    """G005 violates R5 (expired contract) and R9 (PII without consent)."""
    env = ComplianceAuditEnv(task_id="gdpr_privacy_audit")
    env.reset()
    env.step(AuditAction(action_type="inspect_record", record_id="G005"))

    r5 = env.step(AuditAction(action_type="apply_rule", record_id="G005", rule_id="R5"))
    assert r5.reward > 0.0, "R5 should detect expired contract on G005"

    r9 = env.step(AuditAction(action_type="apply_rule", record_id="G005", rule_id="R9"))
    assert r9.reward > 0.0, "R9 should detect missing GDPR consent on G005"
