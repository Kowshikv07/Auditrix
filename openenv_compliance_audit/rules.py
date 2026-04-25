"""Rule engine for the compliance audit environment.

Each rule evaluates a single record (with optional access to all records for
cross-record rules such as duplicate-ID detection).  A rule returns True when
the record *violates* it.

Rules
-----
R1  Minor employee overhours          — age < 18 AND hours > 8
R2  Intern overhours                  — role='intern' AND hours > 40
R3  Salary outside role range         — salary < role_min OR salary > role_max
R4  Duplicate employee ID             — same `id` appears in more than one record
R5  Expired contract still active     — contract_end < audit_date AND status='active'
R6  Background check missing          — sensitive role without background_check=True
R7  Unapproved overtime               — hours > 48 without overtime_approved=True
R8  Missing compliance training       — active employee without compliance_training=True
R9  GDPR consent missing              — pii_access=True without gdpr_consent=True
R10 Missing required fields           — any of {id,name,role,hours,salary} absent/None

Policy overrides
----------------
The EventScheduler can inject mid-episode policy overrides for supported rules:
  - R1: 'minor_hours_threshold' (default 8)
  - R2: 'intern_hours_threshold' (default 40)
  - R7: 'overtime_hours_threshold' (default 48)

Pass a dict like {'R7': {'overtime_hours_threshold': 40}} to evaluate_with_evidence().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Salary ranges per role  (min inclusive, max inclusive)
# ---------------------------------------------------------------------------
ROLE_SALARY_RANGES: Dict[str, Tuple[int, int]] = {
    # Core HR roles
    "employee":        (30_000,   80_000),
    "intern":          (15_000,   35_000),
    "manager":         (60_000,  120_000),
    "director":        (90_000,  180_000),
    "contractor":      (25_000,   90_000),
    # Finance / accounting roles (SOX audit context)
    "finance_manager": (70_000,  130_000),
    "accountant":      (40_000,   90_000),
    "analyst":         (35_000,   75_000),
    "cfo":            (150_000,  350_000),
    # Technology / data roles (GDPR audit context)
    "data_engineer":   (55_000,  110_000),
    "data_analyst":    (40_000,   80_000),
    "ml_engineer":     (70_000,  130_000),
    # Support functions
    "hr":              (40_000,   90_000),
    "security":        (50_000,  100_000),
    # Regulatory / compliance roles (stress-test task)
    "compliance_officer": (55_000, 115_000),
    "legal_counsel":      (80_000, 200_000),
    "vp_finance":         (120_000, 280_000),
    "senior_analyst":     (50_000,  90_000),
    "payroll_specialist": (35_000,  70_000),
}

# Roles that require a completed background check (R6)
BACKGROUND_CHECK_ROLES: Set[str] = {
    "manager", "director", "finance_manager", "accountant",
    "cfo", "security", "hr", "compliance_officer", "legal_counsel",
    "vp_finance",
}

# Violation severity tiers — used by graders for severity-weighted scoring
VIOLATION_SEVERITY: dict = {
    "R1":  "critical",   # child labour — immediate legal liability
    "R4":  "critical",   # data integrity breach — payroll fraud risk
    "R9":  "critical",   # GDPR Art.7 — regulatory fine risk
    "R5":  "high",       # contract governance — legal exposure
    "R6":  "high",       # background check — SOX / security risk
    "R8":  "high",       # SOX §301 training requirement
    "R3":  "medium",     # payroll band — financial control
    "R7":  "medium",     # unapproved overtime — employment law
    "R2":  "low",        # intern hours — advisory
    "R10": "low",        # missing fields — data quality / administrative
    "R11": "high",       # broken manager reference — org integrity
}

SEVERITY_WEIGHTS: dict = {
    "critical": 2.0,
    "high":     1.5,
    "medium":   1.0,
    "low":      0.5,
}

# Salary warning zone: 2% of band width below/above boundary triggers "warning"
# instead of a clean "violation" (ambiguous / near-threshold case)
SALARY_WARNING_PCT = 0.02

# Fixed reference date used for R5 (deterministic, no datetime.today())
AUDIT_REFERENCE_DATE = "2024-01-01"


# ---------------------------------------------------------------------------
# Base rule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    description: str
    condition_summary: str

    def evaluate(
        self,
        record_fields: Dict[str, Any],
        all_record_fields: Optional[List[Dict[str, Any]]] = None,
        policy_override: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return True when the record VIOLATES this rule."""
        raise NotImplementedError

    def evaluate_with_evidence(
        self,
        record_fields: Dict[str, Any],
        all_record_fields: Optional[List[Dict[str, Any]]] = None,
        policy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Return (is_violation, reason_codes, evidence_dict).

        reason_codes: structured list of machine-readable codes like
            ['rule_condition_met', 'field:age=16', 'threshold:hours>8']
        evidence_dict: field-value snapshot relevant to this rule evaluation
        """
        is_violation = self.evaluate(record_fields, all_record_fields, policy_override)
        reason_codes = self._build_reason_codes(record_fields, is_violation, policy_override)
        evidence = self._build_evidence(record_fields, all_record_fields)
        return is_violation, reason_codes, evidence

    def evaluate_with_confidence(
        self,
        record_fields: Dict[str, Any],
        all_record_fields: Optional[List[Dict[str, Any]]] = None,
        policy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """Return (verdict, confidence_tier, reason_codes, evidence).

        verdict: one of "violation" | "warning" | "insufficient_evidence" | "compliant"
          - "violation"              — clear, unambiguous breach
          - "warning"                — near-threshold / ambiguous; flag with caution
          - "insufficient_evidence"  — required field missing; cannot determine
          - "compliant"              — passes cleanly

        confidence_tier: "high" | "medium" | "low"
        """
        # Check for missing required fields first
        evidence = self._build_evidence(record_fields, all_record_fields)
        missing = self._check_missing_fields(record_fields)
        if missing:
            codes = [f"missing_field:{f}" for f in missing] + [f"rule_id:{self.rule_id}"]
            return "insufficient_evidence", "low", codes, evidence

        verdict = self._build_verdict(record_fields, all_record_fields, policy_override)
        is_violation = (verdict in ("violation", "warning"))
        reason_codes = self._build_reason_codes(record_fields, is_violation, policy_override)
        if verdict == "warning":
            reason_codes.append("verdict:near_threshold")
        confidence = "high" if verdict in ("violation", "compliant") else "medium"
        return verdict, confidence, reason_codes, evidence

    def _check_missing_fields(self, record_fields: Dict[str, Any]) -> List[str]:
        """Return list of field names that this rule needs but are missing."""
        return []  # subclasses override

    def _build_verdict(
        self,
        record_fields: Dict[str, Any],
        all_record_fields: Optional[List[Dict[str, Any]]] = None,
        policy_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return verdict string. Default: binary violation/compliant. Subclasses override for warnings."""
        return "violation" if self.evaluate(record_fields, all_record_fields, policy_override) else "compliant"

    def _build_reason_codes(
        self,
        record_fields: Dict[str, Any],
        is_violation: bool,
        policy_override: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Default reason codes — subclasses override for richer output."""
        status = "rule_condition_met" if is_violation else "rule_condition_not_met"
        codes = [status, f"rule_id:{self.rule_id}"]
        if policy_override:
            codes.append("policy_override_active")
        return codes

    def _build_evidence(
        self,
        record_fields: Dict[str, Any],
        all_record_fields: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Original rules (R1 – R5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MinorOverhoursRule(RuleDefinition):
    """R1 — Minor (age < 18) working more than threshold hours per week.

    Threshold defaults to 8. A POLICY_UPDATE event can lower it to e.g. 6.
    """

    DEFAULT_THRESHOLD: int = 8

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        age = record_fields.get("age")
        hours = record_fields.get("hours")
        if age is None or hours is None:
            return False
        threshold = (policy_override or {}).get("minor_hours_threshold", self.DEFAULT_THRESHOLD)
        return int(age) < 18 and float(hours) > threshold

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        threshold = (policy_override or {}).get("minor_hours_threshold", self.DEFAULT_THRESHOLD)
        age = record_fields.get("age")
        hours = record_fields.get("hours")
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R1"]
        if age is not None:
            codes.append(f"field:age={age}")
        if hours is not None:
            codes.append(f"field:hours={hours}")
        codes.append(f"threshold:hours>{threshold}")
        if policy_override and "minor_hours_threshold" in policy_override:
            codes.append(f"policy_override:minor_hours_threshold={threshold}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "age": record_fields.get("age"),
            "hours": record_fields.get("hours"),
            "condition": "age < 18 and hours > threshold",
        }

    def _check_missing_fields(self, record_fields):
        return [f for f in ("age", "hours") if record_fields.get(f) is None]


@dataclass(frozen=True)
class InternOverhoursRule(RuleDefinition):
    """R2 — Intern working more than threshold hours per week.

    Threshold defaults to 40. A POLICY_UPDATE event can change it.
    """

    DEFAULT_THRESHOLD: int = 40

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        role = str(record_fields.get("role", "")).lower()
        hours = record_fields.get("hours")
        if hours is None:
            return False
        threshold = (policy_override or {}).get("intern_hours_threshold", self.DEFAULT_THRESHOLD)
        return role == "intern" and float(hours) > threshold

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        threshold = (policy_override or {}).get("intern_hours_threshold", self.DEFAULT_THRESHOLD)
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R2"]
        codes.append(f"field:role={record_fields.get('role')}")
        codes.append(f"field:hours={record_fields.get('hours')}")
        codes.append(f"threshold:hours>{threshold}")
        if policy_override and "intern_hours_threshold" in policy_override:
            codes.append(f"policy_override:intern_hours_threshold={threshold}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "role": record_fields.get("role"),
            "hours": record_fields.get("hours"),
            "condition": "role == 'intern' and hours > threshold",
        }


@dataclass(frozen=True)
class SalaryRangeRule(RuleDefinition):
    """R3 — Salary outside approved range for the employee's role.

    policy_override keys:
      salary_tolerance_pct — fraction of band width treated as warning zone (default SALARY_WARNING_PCT)
    """

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        role = str(record_fields.get("role", "")).lower()
        salary = record_fields.get("salary")
        if salary is None or role not in ROLE_SALARY_RANGES:
            return False
        lo, hi = ROLE_SALARY_RANGES[role]
        return not (lo <= float(salary) <= hi)

    def _build_verdict(self, record_fields, all_record_fields=None, policy_override=None):
        role = str(record_fields.get("role", "")).lower()
        salary = record_fields.get("salary")
        if salary is None or role not in ROLE_SALARY_RANGES:
            return "compliant"
        lo, hi = ROLE_SALARY_RANGES[role]
        s = float(salary)
        if lo <= s <= hi:
            return "compliant"
        tol_pct = (policy_override or {}).get("salary_tolerance_pct", SALARY_WARNING_PCT)
        warning_zone = tol_pct * (hi - lo)
        if (lo - warning_zone) <= s < lo or hi < s <= (hi + warning_zone):
            return "warning"
        return "violation"

    def _check_missing_fields(self, record_fields):
        return [f for f in ("salary", "role") if record_fields.get(f) is None]

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        role = str(record_fields.get("role", "")).lower()
        salary = record_fields.get("salary")
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R3"]
        codes.append(f"field:role={role}")
        codes.append(f"field:salary={salary}")
        if role in ROLE_SALARY_RANGES:
            lo, hi = ROLE_SALARY_RANGES[role]
            codes.append(f"band:{lo}-{hi}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        role = str(record_fields.get("role", "")).lower()
        salary = record_fields.get("salary")
        band = ROLE_SALARY_RANGES.get(role)
        return {
            "role": role,
            "salary": salary,
            "band_min": band[0] if band else None,
            "band_max": band[1] if band else None,
            "condition": "salary < band_min or salary > band_max",
        }


@dataclass(frozen=True)
class DuplicateIdRule(RuleDefinition):
    """R4 — Employee ID appears more than once in the dataset."""

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        if all_record_fields is None:
            return False
        emp_id = record_fields.get("id")
        if emp_id is None:
            return False
        count = sum(1 for r in all_record_fields if r.get("id") == emp_id)
        return count > 1

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R4"]
        codes.append(f"field:id={record_fields.get('id')}")
        codes.append("cross_record_check_required")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        emp_id = record_fields.get("id")
        count = 0
        if all_record_fields and emp_id is not None:
            count = sum(1 for r in all_record_fields if r.get("id") == emp_id)
        return {
            "id": emp_id,
            "occurrences": count,
            "condition": "id appears more than once",
        }


@dataclass(frozen=True)
class ExpiredContractRule(RuleDefinition):
    """R5 — Contract end date is before the audit reference date but status is 'active'."""

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        status = str(record_fields.get("status", "")).lower()
        contract_end = record_fields.get("contract_end", "")
        if not contract_end or status != "active":
            return False
    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        status = str(record_fields.get("status", "")).lower()
        contract_end = record_fields.get("contract_end", "")
        if not contract_end or status != "active":
            return False
        ref_date = (policy_override or {}).get("audit_reference_date", AUDIT_REFERENCE_DATE)
        return str(contract_end) < ref_date

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        ref_date = (policy_override or {}).get("audit_reference_date", AUDIT_REFERENCE_DATE)
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R5"]
        codes.append(f"field:status={record_fields.get('status')}")
        codes.append(f"field:contract_end={record_fields.get('contract_end')}")
        codes.append(f"reference_date:{ref_date}")
        if policy_override and "audit_reference_date" in policy_override:
            codes.append(f"policy_override:audit_reference_date={ref_date}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "status": record_fields.get("status"),
            "contract_end": record_fields.get("contract_end"),
            "reference_date": AUDIT_REFERENCE_DATE,
            "condition": "contract_end < reference_date and status == 'active'",
        }


# ---------------------------------------------------------------------------
# New real-world rules (R6 – R9)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackgroundCheckRequiredRule(RuleDefinition):
    """R6 — Employees in sensitive roles must have a completed background check.

    policy_override keys:
      additional_sensitive_roles — list[str] of extra roles that now require background checks
        (e.g. a POLICY_UPDATE may add 'contractor' or 'analyst' mid-episode)
    """

    def _effective_sensitive_roles(self, policy_override):
        extra = (policy_override or {}).get("additional_sensitive_roles", [])
        return BACKGROUND_CHECK_ROLES | set(r.lower() for r in extra)

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        role = str(record_fields.get("role", "")).lower()
        if role not in self._effective_sensitive_roles(policy_override):
            return False
        return record_fields.get("background_check") is not True

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        sensitive = self._effective_sensitive_roles(policy_override)
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R6"]
        codes.append(f"field:role={record_fields.get('role')}")
        codes.append(f"field:background_check={record_fields.get('background_check')}")
        codes.append(f"sensitive_role_check:required={record_fields.get('role','').lower() in sensitive}")
        if policy_override and "additional_sensitive_roles" in policy_override:
            codes.append(f"policy_override:additional_sensitive_roles={policy_override['additional_sensitive_roles']}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        role = str(record_fields.get("role", "")).lower()
        return {
            "role": role,
            "background_check": record_fields.get("background_check"),
            "is_sensitive_role": role in BACKGROUND_CHECK_ROLES,
            "condition": "sensitive role and background_check != True",
        }


@dataclass(frozen=True)
class UnapprovedOvertimeRule(RuleDefinition):
    """R7 — Employees working more than threshold hours/week require explicit approval.

    Threshold defaults to 48 (strict >). A POLICY_UPDATE event can lower it.
    Edge case: exactly threshold hours is NOT a violation.
    """

    DEFAULT_THRESHOLD: int = 48

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        hours = record_fields.get("hours")
        if hours is None:
            return False
        threshold = (policy_override or {}).get("overtime_hours_threshold", self.DEFAULT_THRESHOLD)
        if float(hours) <= threshold:
            return False
        return record_fields.get("overtime_approved") is not True

    def _build_verdict(self, record_fields, all_record_fields=None, policy_override=None):
        """hours exactly at threshold → 'warning' (policy-dependent interpretation)."""
        hours = record_fields.get("hours")
        if hours is None:
            return "compliant"
        threshold = (policy_override or {}).get("overtime_hours_threshold", self.DEFAULT_THRESHOLD)
        h = float(hours)
        if h <= threshold - 1:  # clearly below
            return "compliant"
        if h == float(threshold):  # exactly at threshold — ambiguous
            return "warning"
        # above threshold
        return "violation" if record_fields.get("overtime_approved") is not True else "compliant"

    def _check_missing_fields(self, record_fields):
        return [f for f in ("hours",) if record_fields.get(f) is None]

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        threshold = (policy_override or {}).get("overtime_hours_threshold", self.DEFAULT_THRESHOLD)
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R7"]
        codes.append(f"field:hours={record_fields.get('hours')}")
        codes.append(f"field:overtime_approved={record_fields.get('overtime_approved')}")
        codes.append(f"threshold:hours>{threshold}")
        if policy_override and "overtime_hours_threshold" in policy_override:
            codes.append(f"policy_override:overtime_hours_threshold={threshold}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "hours": record_fields.get("hours"),
            "overtime_approved": record_fields.get("overtime_approved"),
            "condition": "hours > threshold and overtime_approved != True",
        }


@dataclass(frozen=True)
class MissingComplianceTrainingRule(RuleDefinition):
    """R8 — All active employees must have completed annual compliance training."""

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        status = str(record_fields.get("status", "active")).lower()
        if status != "active":
            return False
        training_done = record_fields.get("compliance_training")
        return training_done is not True

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R8"]
        codes.append(f"field:status={record_fields.get('status', 'active')}")
        codes.append(f"field:compliance_training={record_fields.get('compliance_training')}")
        codes.append(f"exemption_check:inactive_exempt={record_fields.get('status','active').lower()!='active'}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "status": record_fields.get("status", "active"),
            "compliance_training": record_fields.get("compliance_training"),
            "condition": "status == 'active' and compliance_training != True",
        }


@dataclass(frozen=True)
class GDPRConsentMissingRule(RuleDefinition):
    """R9 — Employees who access PII must have recorded GDPR consent.

    Edge case: employee with pii_access=False is always compliant under this rule.
    """

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        pii_access = bool(record_fields.get("pii_access", False))
        if not pii_access:
            return False
        gdpr_consent = bool(record_fields.get("gdpr_consent", False))
        return not gdpr_consent

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R9"]
        codes.append(f"field:pii_access={record_fields.get('pii_access')}")
        codes.append(f"field:gdpr_consent={record_fields.get('gdpr_consent')}")
        codes.append(f"exemption_check:no_pii_access={not bool(record_fields.get('pii_access', False))}")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        return {
            "pii_access": record_fields.get("pii_access"),
            "gdpr_consent": record_fields.get("gdpr_consent"),
            "condition": "pii_access == True and gdpr_consent != True",
        }


@dataclass(frozen=True)
class MissingRequiredFieldsRule(RuleDefinition):
    """R10 — Record is missing one or more mandatory fields.

    Required fields: id, name, role, hours, salary.
    A field is missing if the key is absent OR the value is None.
    Edge case: a field present with value 0 is NOT missing.
    """

    REQUIRED_FIELDS: tuple = ("id", "name", "role", "hours", "salary")

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        return any(
            record_fields.get(field) is None
            for field in self.REQUIRED_FIELDS
        )

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R10"]
        missing = [f for f in self.REQUIRED_FIELDS if record_fields.get(f) is None]
        for f in missing:
            codes.append(f"missing_field:{f}")
        codes.append("zero_value_valid:True")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        missing = [f for f in self.REQUIRED_FIELDS if record_fields.get(f) is None]
        return {
            "required_fields": list(self.REQUIRED_FIELDS),
            "missing_fields": missing,
            "has_zero_values": [
                f for f in self.REQUIRED_FIELDS
                if f in record_fields and record_fields[f] == 0
            ],
            "condition": "any required field is absent or None",
        }


# ---------------------------------------------------------------------------
# R11 — Manager reference integrity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ManagerReferenceRule(RuleDefinition):
    """R11 — manager_id references a non-existent employee ID in the dataset.

    Requires cross-record access (all_record_fields) to check.
    If manager_id is None the record is exempt (no manager required).
    Forces the agent to maintain a mental model of which employee IDs exist.
    """

    def evaluate(self, record_fields, all_record_fields=None, policy_override=None):
        manager_id = record_fields.get("manager_id")
        if manager_id is None:
            return False
        if all_record_fields is None:
            return False  # can't cross-check without dataset context
        existing_ids = {r.get("id") for r in all_record_fields if r.get("id") is not None}
        return manager_id not in existing_ids

    def _build_reason_codes(self, record_fields, is_violation, policy_override):
        codes = ["rule_condition_met" if is_violation else "rule_condition_not_met", "rule_id:R11"]
        codes.append(f"field:manager_id={record_fields.get('manager_id')}")
        codes.append("cross_record_check_required")
        return codes

    def _build_evidence(self, record_fields, all_record_fields=None):
        manager_id = record_fields.get("manager_id")
        existing = []
        if all_record_fields and manager_id is not None:
            existing = [r.get("id") for r in all_record_fields if r.get("id") is not None]
        return {
            "manager_id": manager_id,
            "manager_exists": (manager_id in existing) if manager_id is not None else None,
            "condition": "manager_id is not None and manager_id not in {all employee ids}",
        }


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------
RULES: Dict[str, RuleDefinition] = {
    "R1": MinorOverhoursRule(
        rule_id="R1",
        description="Minor employee overhours",
        condition_summary="age < 18 and hours > 8",
    ),
    "R2": InternOverhoursRule(
        rule_id="R2",
        description="Intern overhours",
        condition_summary="role == 'intern' and hours > 40",
    ),
    "R3": SalaryRangeRule(
        rule_id="R3",
        description="Salary outside role range",
        condition_summary="salary < role_min_salary or salary > role_max_salary",
    ),
    "R4": DuplicateIdRule(
        rule_id="R4",
        description="Duplicate employee ID",
        condition_summary="employee id appears more than once in the dataset",
    ),
    "R5": ExpiredContractRule(
        rule_id="R5",
        description="Expired contract still active",
        condition_summary="contract_end < '2024-01-01' and status == 'active'",
    ),
    "R6": BackgroundCheckRequiredRule(
        rule_id="R6",
        description="Background check missing for sensitive role",
        condition_summary=(
            "role in {manager, director, finance_manager, accountant, cfo, security, hr, "
            "compliance_officer, legal_counsel, vp_finance} "
            "and background_check != True"
        ),
    ),
    "R7": UnapprovedOvertimeRule(
        rule_id="R7",
        description="Unapproved overtime (>48 h/week)",
        condition_summary="hours > 48 and overtime_approved != True",
    ),
    "R8": MissingComplianceTrainingRule(
        rule_id="R8",
        description="Missing annual compliance training",
        condition_summary="status == 'active' and compliance_training != True",
    ),
    "R9": GDPRConsentMissingRule(
        rule_id="R9",
        description="GDPR consent missing for PII data access",
        condition_summary="pii_access == True and gdpr_consent != True",
    ),
    "R10": MissingRequiredFieldsRule(
        rule_id="R10",
        description="Missing required fields (incomplete record)",
        condition_summary=(
            "any of [id, name, role, hours, salary] is absent or None"
        ),
    ),
    "R11": ManagerReferenceRule(
        rule_id="R11",
        description="Manager ID references non-existent employee",
        condition_summary="manager_id is not None and manager_id not in {all employee ids}",
    ),
}


def rule_info() -> List[Dict[str, str]]:
    """Return serialisable rule metadata (for observations), including severity tier."""
    return [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "condition": r.condition_summary,
            "severity": VIOLATION_SEVERITY.get(r.rule_id, "medium"),
        }
        for r in RULES.values()
    ]
