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
}

# Roles that require a completed background check (R6)
BACKGROUND_CHECK_ROLES: Set[str] = {
    "manager", "director", "finance_manager", "accountant",
    "cfo", "security", "hr",
}

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
    ) -> bool:
        """Return True when the record VIOLATES this rule."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Original rules (R1 – R5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MinorOverhoursRule(RuleDefinition):
    """R1 — Minor (age < 18) working more than 8 hours per week."""

    def evaluate(self, record_fields, all_record_fields=None):
        age = record_fields.get("age")
        hours = record_fields.get("hours")
        if age is None or hours is None:
            return False
        return int(age) < 18 and float(hours) > 8


@dataclass(frozen=True)
class InternOverhoursRule(RuleDefinition):
    """R2 — Intern working more than 40 hours per week."""

    def evaluate(self, record_fields, all_record_fields=None):
        role = str(record_fields.get("role", "")).lower()
        hours = record_fields.get("hours")
        if hours is None:
            return False
        return role == "intern" and float(hours) > 40


@dataclass(frozen=True)
class SalaryRangeRule(RuleDefinition):
    """R3 — Salary outside approved range for the employee's role."""

    def evaluate(self, record_fields, all_record_fields=None):
        role = str(record_fields.get("role", "")).lower()
        salary = record_fields.get("salary")
        if salary is None or role not in ROLE_SALARY_RANGES:
            return False
        lo, hi = ROLE_SALARY_RANGES[role]
        return not (lo <= float(salary) <= hi)


@dataclass(frozen=True)
class DuplicateIdRule(RuleDefinition):
    """R4 — Employee ID appears more than once in the dataset."""

    def evaluate(self, record_fields, all_record_fields=None):
        if all_record_fields is None:
            return False
        emp_id = record_fields.get("id")
        if emp_id is None:
            return False
        count = sum(1 for r in all_record_fields if r.get("id") == emp_id)
        return count > 1


@dataclass(frozen=True)
class ExpiredContractRule(RuleDefinition):
    """R5 — Contract end date is before the audit reference date but status is 'active'."""

    def evaluate(self, record_fields, all_record_fields=None):
        status = str(record_fields.get("status", "")).lower()
        contract_end = record_fields.get("contract_end", "")
        if not contract_end or status != "active":
            return False
        # Lexicographic comparison works for ISO-8601 dates (YYYY-MM-DD)
        return str(contract_end) < AUDIT_REFERENCE_DATE


# ---------------------------------------------------------------------------
# New real-world rules (R6 – R9)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackgroundCheckRequiredRule(RuleDefinition):
    """R6 — Employees in sensitive roles must have a completed background check.

    Sensitive roles include managers, directors, finance roles, HR, and security
    personnel who have access to confidential data or financial systems.
    Typical in SOX, ISO 27001, and internal HR policy compliance frameworks.
    """

    def evaluate(self, record_fields, all_record_fields=None):
        role = str(record_fields.get("role", "")).lower()
        if role not in BACKGROUND_CHECK_ROLES:
            return False  # Role does not require a background check
        bg_check = record_fields.get("background_check")
        # background_check must be explicitly True; None / False / missing = violation
        return bg_check is not True


@dataclass(frozen=True)
class UnapprovedOvertimeRule(RuleDefinition):
    """R7 — Employees working more than 48 hours/week require explicit overtime approval.

    Many jurisdictions (EU Working Time Directive, US FLSA) require documented
    approval for sustained overtime above 48 h/week.  Without overtime_approved=True
    the record is a labour-law violation.
    Edge case: exactly 48 hours is NOT a violation (strict > check).
    """

    def evaluate(self, record_fields, all_record_fields=None):
        hours = record_fields.get("hours")
        if hours is None:
            return False
        if float(hours) <= 48:
            return False  # 48 h or under: no approval required
        # Above 48 h: overtime_approved must be explicitly True
        return record_fields.get("overtime_approved") is not True


@dataclass(frozen=True)
class MissingComplianceTrainingRule(RuleDefinition):
    """R8 — All active employees must have completed annual compliance training.

    Mandatory under SOX Section 301, GDPR Article 39, and most corporate
    governance frameworks.  Inactive / terminated employees are exempt.
    The field compliance_training must be True; missing or False = violation.
    """

    def evaluate(self, record_fields, all_record_fields=None):
        # Only active employees are required to have completed training
        status = str(record_fields.get("status", "active")).lower()
        if status != "active":
            return False
        training_done = record_fields.get("compliance_training")
        return training_done is not True


@dataclass(frozen=True)
class GDPRConsentMissingRule(RuleDefinition):
    """R9 — Employees who access personal identifiable information (PII) must have
    a recorded GDPR / data-processing consent on file.

    Required under GDPR Article 7 and equivalent legislation (CCPA, PDPA).
    If pii_access is False or absent the rule does not apply.
    gdpr_consent must be explicitly True; missing or False = violation.
    Edge case: employee with pii_access=False is always compliant under this rule.
    """

    def evaluate(self, record_fields, all_record_fields=None):
        pii_access = bool(record_fields.get("pii_access", False))
        if not pii_access:
            return False  # No PII access — consent not required
        gdpr_consent = bool(record_fields.get("gdpr_consent", False))
        return not gdpr_consent


@dataclass(frozen=True)
class MissingRequiredFieldsRule(RuleDefinition):
    """R10 — Record is missing one or more mandatory fields.

    In any HR or payroll system, a record must have all core fields populated
    to be processable.  Missing fields indicate incomplete data entry, which
    causes downstream errors in payroll runs, tax filings, and compliance
    reports (SOX, GDPR, ISO 27001 all require data completeness).

    Required fields: id, name, role, hours, salary.
    A field is considered missing if the key is absent OR the value is None.

    Edge case: a field present with value 0 (e.g. hours=0 for inactive staff)
    is NOT missing — only None / absent counts as a data gap.
    """

    REQUIRED_FIELDS: tuple = ("id", "name", "role", "hours", "salary")

    def evaluate(self, record_fields, all_record_fields=None):
        return any(
            record_fields.get(field) is None
            for field in self.REQUIRED_FIELDS
        )


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
            "role in {manager, director, finance_manager, accountant, cfo, security, hr} "
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
}


def rule_info() -> List[Dict[str, str]]:
    """Return serialisable rule metadata (for observations)."""
    return [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "condition": r.condition_summary,
        }
        for r in RULES.values()
    ]
