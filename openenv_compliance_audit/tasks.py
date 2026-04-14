"""Task definitions for the compliance audit environment.

Five tasks across three difficulty levels:

  easy   — 5 records,  2 rules (R1, R2),          2 clear violations
  medium — 12 records, 4 rules (R1-R4),            9 violations; hidden & overlapping
    hard   — 20 records, 5 rules (R1-R5),            15 violation pairs; edge cases
  ── NEW real-world tasks ──────────────────────────────────────────────────────
  finance_sox_audit  — 15 finance-dept records, rules R3/R6/R7/R8 (SOX-style)
                       Salary overrides, missing background checks, unapproved
                       overtime, and absent compliance training. Difficulty: hard.
  gdpr_privacy_audit — 10 data-team records, rules R5/R8/R9 (GDPR-style)
                       Expired contracts, missing training, and PII data access
                       without recorded consent. Difficulty: medium.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class RecordTruth:
    """Ground-truth definition of one organisational record."""

    record_id: str
    fields: Dict[str, Any]
    # rule_ids that should be flagged as violations for this record
    expected_violations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard", "extreme"]
    objective: str
    max_steps: int
    active_rule_ids: List[str]
    records: List[RecordTruth]


# ---------------------------------------------------------------------------
# Easy — 5 records, rules R1 + R2
# 2 violations: E001 (R1), E002 (R2)
# ---------------------------------------------------------------------------
_EASY_RECORDS: List[RecordTruth] = [
    RecordTruth(
        record_id="E001",
        fields={"id": 1, "name": "Alex Turner", "age": 17, "role": "employee",
                "hours": 10, "salary": 25000},
        expected_violations=["R1"],   # minor working > 8 h
    ),
    RecordTruth(
        record_id="E002",
        fields={"id": 2, "name": "Sam Rivera", "age": 25, "role": "intern",
                "hours": 45, "salary": 22000},
        expected_violations=["R2"],   # intern working > 40 h
    ),
    RecordTruth(
        record_id="E003",
        fields={"id": 3, "name": "Jordan Lee", "age": 30, "role": "employee",
                "hours": 40, "salary": 55000},
        expected_violations=[],       # compliant
    ),
    RecordTruth(
        record_id="E004",
        fields={"id": 4, "name": "Morgan Kim", "age": 22, "role": "manager",
                "hours": 35, "salary": 75000},
        expected_violations=[],       # compliant
    ),
    RecordTruth(
        record_id="E005",
        fields={"id": 5, "name": "Casey Patel", "age": 16, "role": "employee",
                "hours": 7, "salary": 20000},
        expected_violations=[],       # hours = 7 ≤ 8 → R1 does NOT trigger
    ),
]

# ---------------------------------------------------------------------------
# Medium — 12 records, rules R1 + R2 + R3 + R4
# 9 violations: M001(R1), M002(R2), M003(R3), M006(R3),
#               M007(R4), M008(R4), M009(R1), M010(R3), M012(R3)
# ---------------------------------------------------------------------------
_MEDIUM_RECORDS: List[RecordTruth] = [
    RecordTruth(
        record_id="M001",
        fields={"id": 1, "name": "Dana Fox",    "age": 16, "role": "employee",
                "hours": 9,  "salary": 35000},
        expected_violations=["R1"],
    ),
    RecordTruth(
        record_id="M002",
        fields={"id": 2, "name": "Eli Cross",   "age": 24, "role": "intern",
                "hours": 42, "salary": 20000},
        expected_violations=["R2"],
    ),
    RecordTruth(
        record_id="M003",
        fields={"id": 3, "name": "Fiona Grant",  "age": 35, "role": "manager",
                "hours": 38, "salary": 50000},    # manager min = 60 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="M004",
        fields={"id": 4, "name": "Glen Hart",    "age": 28, "role": "employee",
                "hours": 40, "salary": 55000},
        expected_violations=[],
    ),
    RecordTruth(
        record_id="M005",
        fields={"id": 5, "name": "Hana Imai",    "age": 22, "role": "intern",
                "hours": 38, "salary": 18000},
        expected_violations=[],
    ),
    RecordTruth(
        record_id="M006",
        fields={"id": 6, "name": "Ivan Joris",   "age": 45, "role": "director",
                "hours": 35, "salary": 200000},   # director max = 180 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="M007",
        fields={"id": 7, "name": "Julia Kane",   "age": 31, "role": "employee",
                "hours": 35, "salary": 65000},    # id=7 duplicated with M008
        expected_violations=["R4"],
    ),
    RecordTruth(
        record_id="M008",
        fields={"id": 7, "name": "Karl Lund",    "age": 29, "role": "employee",
                "hours": 40, "salary": 60000},    # id=7 duplicated with M007
        expected_violations=["R4"],
    ),
    RecordTruth(
        record_id="M009",
        fields={"id": 9, "name": "Lara Moss",    "age": 17, "role": "intern",
                "hours": 38, "salary": 15000},    # age<18 and hours=38>8 → R1; hours≤40 → R2 OK
        expected_violations=["R1"],
    ),
    RecordTruth(
        record_id="M010",
        fields={"id": 10, "name": "Mia Nash",   "age": 29, "role": "contractor",
                "hours": 45, "salary": 95000},    # contractor max = 90 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="M011",
        fields={"id": 11, "name": "Noah Osei",  "age": 33, "role": "employee",
                "hours": 45, "salary": 70000},
        expected_violations=[],
    ),
    RecordTruth(
        record_id="M012",
        fields={"id": 12, "name": "Olivia Park", "age": 19, "role": "employee",
                "hours": 35, "salary": 25000},    # employee min = 30 000
        expected_violations=["R3"],
    ),
]

# ---------------------------------------------------------------------------
# Hard — 20 records, rules R1 – R5
# 16 (record_id, rule_id) violation pairs with overlapping and edge cases
# ---------------------------------------------------------------------------
_HARD_RECORDS: List[RecordTruth] = [
    RecordTruth(
        record_id="H001",
        fields={"id": 1,  "name": "Aaron Blake",  "age": 16, "role": "employee",
                "hours": 10, "salary": 45000, "status": "active",
                "contract_end": "2025-06-01"},
        expected_violations=["R1"],
    ),
    RecordTruth(
        record_id="H002",
        fields={"id": 2,  "name": "Bella Chase",  "age": 24, "role": "intern",
                "hours": 45, "salary": 18000, "status": "active",
                "contract_end": "2025-06-01"},
        expected_violations=["R2"],
    ),
    RecordTruth(
        record_id="H003",
        fields={"id": 3,  "name": "Caleb Drake",  "age": 35, "role": "manager",
                "hours": 38, "salary": 55000, "status": "active",
                "contract_end": "2025-06-01"},   # manager min = 60 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="H004",
        fields={"id": 4,  "name": "Diana Evans",  "age": 22, "role": "director",
                "hours": 35, "salary": 200000, "status": "active",
                "contract_end": "2025-06-01"},   # director max = 180 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="H005",
        fields={"id": 5,  "name": "Ethan Ford",   "age": 17, "role": "intern",
                "hours": 42, "salary": 16000, "status": "active",
                "contract_end": "2025-06-01"},   # R1 (age<18,hours>8) + R2 (intern,hours>40)
        expected_violations=["R1", "R2"],
    ),
    RecordTruth(
        record_id="H006",
        fields={"id": 6,  "name": "Faye Green",   "age": 28, "role": "employee",
                "hours": 40, "salary": 60000, "status": "active",
                "contract_end": "2025-06-01"},   # id=6 duplicated with H007
        expected_violations=["R4"],
    ),
    RecordTruth(
        record_id="H007",
        fields={"id": 6,  "name": "George Hill",  "age": 31, "role": "employee",
                "hours": 38, "salary": 65000, "status": "active",
                "contract_end": "2025-06-01"},   # id=6 duplicated with H006
        expected_violations=["R4"],
    ),
    RecordTruth(
        record_id="H008",
        fields={"id": 8,  "name": "Hannah Ivers", "age": 18, "role": "employee",
                "hours": 10, "salary": 35000, "status": "active",
                "contract_end": "2025-06-01"},   # age=18 (NOT < 18) → R1 edge case: compliant
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H009",
        fields={"id": 9,  "name": "Isaac Jones",  "age": 29, "role": "contractor",
                "hours": 45, "salary": 95000, "status": "active",
                "contract_end": "2025-06-01"},   # contractor max = 90 000
        expected_violations=["R3"],
    ),
    RecordTruth(
        record_id="H010",
        fields={"id": 10, "name": "Julia King",   "age": 26, "role": "employee",
                "hours": 40, "salary": 80000, "status": "active",
                "contract_end": "2025-06-01"},   # exactly at max → compliant
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H011",
        fields={"id": 11, "name": "Kevin Lane",   "age": 32, "role": "manager",
                "hours": 38, "salary": 120000, "status": "active",
                "contract_end": "2025-06-01"},   # exactly at max → compliant
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H012",
        fields={"id": 12, "name": "Laura Marsh",  "age": 30, "role": "employee",
                "hours": 40, "salary": 55000, "status": "active",
                "contract_end": "2023-06-01"},   # expired contract → R5
        expected_violations=["R5"],
    ),
    RecordTruth(
        record_id="H013",
        fields={"id": 13, "name": "Marcus Nash",  "age": 27, "role": "contractor",
                "hours": 35, "salary": 45000, "status": "active",
                "contract_end": "2023-12-31"},   # expired contract → R5
        expected_violations=["R5"],
    ),
    RecordTruth(
        record_id="H014",
        fields={"id": 14, "name": "Nora Owen",    "age": 29, "role": "employee",
                "hours": 40, "salary": 60000, "status": "inactive",
                "contract_end": "2023-06-01"},   # status != "active" → R5 doesn't trigger
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H015",
        fields={"id": 15, "name": "Oscar Price",  "age": 15, "role": "employee",
                "hours": 8,  "salary": 30000, "status": "active",
                "contract_end": "2025-06-01"},   # hours=8 (NOT > 8) → R1 edge case: compliant
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H016",
        fields={"id": 16, "name": "Paula Quinn",  "age": 16, "role": "employee",
                "hours": 9,  "salary": 30000, "status": "active",
                "contract_end": "2025-06-01"},
        expected_violations=["R1"],
    ),
    RecordTruth(
        record_id="H017",
        fields={"id": 17, "name": "Quinn Reid",   "age": 22, "role": "intern",
                "hours": 40, "salary": 22000, "status": "active",
                "contract_end": "2025-06-01"},   # hours=40 (NOT > 40) → R2 edge case: compliant
        expected_violations=[],
    ),
    RecordTruth(
        record_id="H018",
        fields={"id": 18, "name": "Rachel Scott",  "age": 23, "role": "intern",
                "hours": 41, "salary": 22000, "status": "active",
                "contract_end": "2025-06-01"},
        expected_violations=["R2"],
    ),
    RecordTruth(
        record_id="H019",
        fields={"id": 19, "name": "Samuel Todd",  "age": 28, "role": "employee",
                "hours": 35, "salary": 70000, "status": "active",
                "contract_end": "2025-06-01"},   # id=19 duplicated with H020
        expected_violations=["R4"],
    ),
    RecordTruth(
        record_id="H020",
        fields={"id": 19, "name": "Tina Upton",   "age": 33, "role": "manager",
                "hours": 40, "salary": 90000, "status": "active",
                "contract_end": "2025-06-01"},   # id=19 duplicated with H019; salary 90000 in manager range
        expected_violations=["R4"],
    ),
]


# ============================================================================
# NEW TASK 1: Finance / SOX Compliance Audit  (Hard, 15 records, R3+R6+R7+R8)
# ============================================================================
#
# Scenario
# --------
# The internal audit team is reviewing the Finance department ahead of the
# annual SOX (Sarbanes-Oxley) certification.  Every active employee must:
#   • Have a salary within the band for their role              (R3)
#   • Have a completed background check (sensitive-role policy) (R6)
#   • Not be working >48 h/week without documented approval     (R7)
#   • Have completed the annual compliance training             (R8)
#
# Violation summary (17 (record, rule) pairs):
#   F001: R7          F002: R6          F003: R8          F004: R7
#   F005: R3          F006: R3          F007: R3+R8       F008: —
#   F009: R3+R7       F010: R3+R6       F011: R7+R8       F012: —
#   F013: R3          F014: —           F015: R3+R8
# ============================================================================
_FINANCE_SOX_RECORDS: List[RecordTruth] = [
    # F001: hours=52, no overtime approval → R7
    RecordTruth(
        record_id="F001",
        fields={
            "id": 101, "name": "Richard Holt",  "age": 42, "role": "finance_manager",
            "department": "Treasury", "salary": 95000, "hours": 52,
            "status": "active", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
        },
        expected_violations=["R7"],
    ),
    # F002: accountant without background check → R6
    RecordTruth(
        record_id="F002",
        fields={
            "id": 102, "name": "Priya Sharma",  "age": 29, "role": "accountant",
            "department": "Accounts Payable", "salary": 42000, "hours": 40,
            "status": "active", "background_check": False,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=["R6"],
    ),
    # F003: director missing compliance training → R8
    RecordTruth(
        record_id="F003",
        fields={
            "id": 103, "name": "James Okafor",  "age": 51, "role": "director",
            "department": "Financial Reporting", "salary": 145000, "hours": 45,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": False,
        },
        expected_violations=["R8"],
    ),
    # F004: regular employee, 55 h, no overtime approval → R7
    RecordTruth(
        record_id="F004",
        fields={
            "id": 104, "name": "Chen Wei",       "age": 34, "role": "employee",
            "department": "Reconciliation", "salary": 62000, "hours": 55,
            "status": "active", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
        },
        expected_violations=["R7"],
    ),
    # F005: manager salary 125 000 (max 120 000) → R3
    RecordTruth(
        record_id="F005",
        fields={
            "id": 105, "name": "Amara Diallo",   "age": 38, "role": "manager",
            "department": "Internal Controls", "salary": 125000, "hours": 40,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=["R3"],
    ),
    # F006: analyst salary 80 000 (max 75 000) → R3
    RecordTruth(
        record_id="F006",
        fields={
            "id": 106, "name": "Lena Fischer",   "age": 27, "role": "analyst",
            "department": "FP&A", "salary": 80000, "hours": 40,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=["R3"],
    ),
    # F007: finance_manager salary 140 000 (max 130 000) → R3; also missing training → R8
    RecordTruth(
        record_id="F007",
        fields={
            "id": 107, "name": "Bruno Ferreira", "age": 44, "role": "finance_manager",
            "department": "Treasury", "salary": 140000, "hours": 42,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": False,
        },
        expected_violations=["R3", "R8"],
    ),
    # F008: fully compliant employee
    RecordTruth(
        record_id="F008",
        fields={
            "id": 108, "name": "Sofia Kovacs",   "age": 31, "role": "employee",
            "department": "Billing", "salary": 60000, "hours": 42,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=[],
    ),
    # F009: contractor 50h, no OT approval (R7); salary 92 000 > 90 000 max (R3)
    RecordTruth(
        record_id="F009",
        fields={
            "id": 109, "name": "Derek Walls",    "age": 36, "role": "contractor",
            "department": "Tax Advisory", "salary": 92000, "hours": 50,
            "status": "active", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
        },
        expected_violations=["R3", "R7"],
    ),
    # F010: director salary 190 000 > 180 000 (R3); missing background check (R6)
    RecordTruth(
        record_id="F010",
        fields={
            "id": 110, "name": "Yuki Tanaka",    "age": 55, "role": "director",
            "department": "Corporate Finance", "salary": 190000, "hours": 45,
            "status": "active", "background_check": False,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=["R3", "R6"],
    ),
    # F011: accountant 55h no OT approval (R7); missing training (R8)
    RecordTruth(
        record_id="F011",
        fields={
            "id": 111, "name": "Marie Dupont",   "age": 28, "role": "accountant",
            "department": "General Ledger", "salary": 50000, "hours": 55,
            "status": "active", "background_check": True,
            "overtime_approved": False, "compliance_training": False,
        },
        expected_violations=["R7", "R8"],
    ),
    # F012: fully compliant manager
    RecordTruth(
        record_id="F012",
        fields={
            "id": 112, "name": "Carlos Ruiz",    "age": 40, "role": "manager",
            "department": "Payroll", "salary": 95000, "hours": 38,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=[],
    ),
    # F013: finance_manager salary 65 000 < 70 000 min → R3
    RecordTruth(
        record_id="F013",
        fields={
            "id": 113, "name": "Fatou Ba",       "age": 32, "role": "finance_manager",
            "department": "Budgeting", "salary": 65000, "hours": 38,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=["R3"],
    ),
    # F014: fully compliant employee
    RecordTruth(
        record_id="F014",
        fields={
            "id": 114, "name": "Tom Nguyen",     "age": 25, "role": "employee",
            "department": "Collections", "salary": 45000, "hours": 35,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
        },
        expected_violations=[],
    ),
    # F015: analyst salary 32 000 < 35 000 min (R3); hours=48 (NOT > 48, edge!) → no R7;
    #        missing training (R8)
    RecordTruth(
        record_id="F015",
        fields={
            "id": 115, "name": "Anya Ivanova",   "age": 24, "role": "analyst",
            "department": "FP&A", "salary": 32000, "hours": 48,
            "status": "active", "background_check": True,
            "overtime_approved": True, "compliance_training": False,
        },
        expected_violations=["R3", "R8"],   # hours=48 is NOT > 48 → R7 does NOT trigger
    ),
]


# ============================================================================
# NEW TASK 2: GDPR Data-Privacy Compliance Audit  (Medium, 10 records, R5+R8+R9)
# ============================================================================
#
# Scenario
# --------
# The Data Protection Officer (DPO) is running a quarterly GDPR audit on the
# Engineering & Analytics team.  Every active employee must:
#   • Have a valid (non-expired) contract on file                (R5)
#   • Have completed the annual data-privacy training            (R8)
#   • Have GDPR consent recorded if they access PII data         (R9)
#
# The challenge: some records are inactive (R5 and R8 don't apply), and some
# employees do not handle PII at all (R9 doesn't apply).  The agent must
# reason about field presence and values to avoid false positives.
#
# Violation summary (9 (record, rule) pairs):
#   G001: —         G002: R9           G003: R8
#   G004: —         G005: R5+R9        G006: —
#   G007: R8        G008: —            G009: R5+R8       G010: R8+R9
# ============================================================================
_GDPR_PRIVACY_RECORDS: List[RecordTruth] = [
    # G001: fully compliant data engineer with PII access + consent + training
    RecordTruth(
        record_id="G001",
        fields={
            "id": 201, "name": "Alice Morin",    "age": 30, "role": "data_engineer",
            "department": "Platform Engineering", "salary": 88000, "hours": 40,
            "status": "active", "contract_end": "2025-12-01",
            "pii_access": True,  "gdpr_consent": True,  "compliance_training": True,
        },
        expected_violations=[],
    ),
    # G002: analyst has PII access but no GDPR consent on file → R9
    RecordTruth(
        record_id="G002",
        fields={
            "id": 202, "name": "Ben Osei",       "age": 26, "role": "analyst",
            "department": "Customer Insights", "salary": 55000, "hours": 38,
            "status": "active", "contract_end": "2025-06-01",
            "pii_access": True,  "gdpr_consent": False, "compliance_training": True,
        },
        expected_violations=["R9"],
    ),
    # G003: data engineer, no PII access but missing training → R8
    RecordTruth(
        record_id="G003",
        fields={
            "id": 203, "name": "Clara Nkosi",    "age": 29, "role": "data_engineer",
            "department": "Data Infrastructure", "salary": 92000, "hours": 40,
            "status": "active", "contract_end": "2025-09-01",
            "pii_access": False, "gdpr_consent": False, "compliance_training": False,
        },
        expected_violations=["R8"],   # no pii_access → R9 not applicable
    ),
    # G004: fully compliant manager with PII access
    RecordTruth(
        record_id="G004",
        fields={
            "id": 204, "name": "David Chen",     "age": 38, "role": "manager",
            "department": "Data Analytics", "salary": 105000, "hours": 45,
            "status": "active", "contract_end": "2026-03-01",
            "pii_access": True,  "gdpr_consent": True,  "compliance_training": True,
        },
        expected_violations=[],
    ),
    # G005: contractor — expired contract (R5) + PII access without consent (R9)
    RecordTruth(
        record_id="G005",
        fields={
            "id": 205, "name": "Elise Bouchard", "age": 33, "role": "contractor",
            "department": "ML Platform", "salary": 78000, "hours": 40,
            "status": "active", "contract_end": "2023-09-01",   # expired before audit date
            "pii_access": True,  "gdpr_consent": False, "compliance_training": True,
        },
        expected_violations=["R5", "R9"],
    ),
    # G006: analyst, no PII access, training done, valid contract — fully compliant
    RecordTruth(
        record_id="G006",
        fields={
            "id": 206, "name": "Frank Mueller",  "age": 24, "role": "analyst",
            "department": "Business Intelligence", "salary": 48000, "hours": 38,
            "status": "active", "contract_end": "2025-06-01",
            "pii_access": False, "gdpr_consent": False, "compliance_training": True,
        },
        expected_violations=[],   # no pii_access → R9 doesn't apply; training done; contract valid
    ),
    # G007: ML engineer with PII access + consent but missing compliance training → R8
    RecordTruth(
        record_id="G007",
        fields={
            "id": 207, "name": "Gina Park",      "age": 31, "role": "ml_engineer",
            "department": "Recommendations", "salary": 112000, "hours": 40,
            "status": "active", "contract_end": "2025-11-01",
            "pii_access": True,  "gdpr_consent": True,  "compliance_training": False,
        },
        expected_violations=["R8"],
    ),
    # G008: INACTIVE employee — expired contract and no training but status=inactive,
    #        so R5 and R8 do NOT apply. PII access + consent → R9 OK too.
    #        Edge case: agent must not flag inactive employees.
    RecordTruth(
        record_id="G008",
        fields={
            "id": 208, "name": "Hector Vega",    "age": 45, "role": "employee",
            "department": "Legacy Systems", "salary": 70000, "hours": 0,
            "status": "inactive", "contract_end": "2023-01-01",
            "pii_access": True,  "gdpr_consent": True,  "compliance_training": False,
        },
        expected_violations=[],   # inactive → neither R5 nor R8 applies; consent OK → R9 OK
    ),
    # G009: active contractor — contract expired (R5) + no training (R8);
    #        no PII access → R9 doesn't apply
    RecordTruth(
        record_id="G009",
        fields={
            "id": 209, "name": "Iris Tanaka",    "age": 27, "role": "contractor",
            "department": "Data Quality", "salary": 65000, "hours": 40,
            "status": "active", "contract_end": "2023-11-01",   # expired
            "pii_access": False, "gdpr_consent": False, "compliance_training": False,
        },
        expected_violations=["R5", "R8"],
    ),
    # G010: analyst — active, PII access, no consent (R9) + no training (R8)
    RecordTruth(
        record_id="G010",
        fields={
            "id": 210, "name": "Jordan Obi",     "age": 28, "role": "analyst",
            "department": "Customer Analytics", "salary": 58000, "hours": 40,
            "status": "active", "contract_end": "2025-08-01",
            "pii_access": True,  "gdpr_consent": False, "compliance_training": False,
        },
        expected_violations=["R8", "R9"],
    ),
]


# ---------------------------------------------------------------------------
# ============================================================================
# NEW TASK 3: HR Data Integrity Audit  (Medium, 8 records, R3+R4+R10)
# ============================================================================
#
# Scenario
# --------
# The HR operations team found inconsistencies during a payroll pre-run.
# An audit is needed to identify records with:
#   • Salary outside the approved band for their role         (R3)
#   • Duplicate employee IDs (causes double-payroll entries)  (R4)
#   • Missing mandatory fields (blocks payroll processing)    (R10)
#
# This directly mirrors the "Data Integrity" rule category in the problem spec.
# The tricky part: a record MISSING salary gracefully skips R3 (null-safe),
# but STILL triggers R10 — so the agent must apply both rules independently.
#
# Violation summary (6 (record, rule) pairs):
#   DI001: —          DI002: R3          DI003: R10
#   DI004: R4         DI005: R4          DI006: R10
#   DI007: —          DI008: R10
# ============================================================================
_DATA_INTEGRITY_RECORDS: List[RecordTruth] = [
    # DI001: fully complete and compliant record
    RecordTruth(
        record_id="DI001",
        fields={"id": 1, "name": "Alice Brown",   "role": "employee",
                "hours": 40, "salary": 60000},
        expected_violations=[],
    ),
    # DI002: manager salary 130 000 (max 120 000) → R3; all fields present
    RecordTruth(
        record_id="DI002",
        fields={"id": 2, "name": "Bob Chan",      "role": "manager",
                "hours": 38, "salary": 130000},
        expected_violations=["R3"],
    ),
    # DI003: salary field missing entirely → R10; R3 skips gracefully (salary=None)
    RecordTruth(
        record_id="DI003",
        fields={"id": 3, "name": "Carol Davis",   "role": "employee",
                "hours": 35},          # salary deliberately omitted
        expected_violations=["R10"],
    ),
    # DI004: id=4 duplicated with DI005 → R4; all fields complete
    RecordTruth(
        record_id="DI004",
        fields={"id": 4, "name": "David Evans",   "role": "intern",
                "hours": 35, "salary": 20000},
        expected_violations=["R4"],
    ),
    # DI005: id=4 duplicated with DI004 → R4; all fields complete
    RecordTruth(
        record_id="DI005",
        fields={"id": 4, "name": "Eve Franklin",  "role": "intern",
                "hours": 38, "salary": 18000},
        expected_violations=["R4"],
    ),
    # DI006: hours field missing → R10; role is employee so R3 would need salary (present=55000)
    #        but hours=None doesn't affect R3 (which only checks salary) → only R10 fires
    RecordTruth(
        record_id="DI006",
        fields={"id": 6, "name": "Frank Green",   "role": "employee",
                "salary": 55000},      # hours deliberately omitted
        expected_violations=["R10"],
    ),
    # DI007: fully complete and compliant record
    RecordTruth(
        record_id="DI007",
        fields={"id": 7, "name": "Grace Hall",    "role": "employee",
                "hours": 40, "salary": 50000},
        expected_violations=[],
    ),
    # DI008: role field missing → R10; also salary present but R3 needs role → skips gracefully
    #        Edge case: hours=0 is valid (not missing) — only role is absent
    RecordTruth(
        record_id="DI008",
        fields={"id": 8, "name": "Hank Irons",
                "hours": 40, "salary": 55000},    # role deliberately omitted
        expected_violations=["R10"],
    ),
]


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


# ============================================================================
# NEW TASK 4: Regulatory Storm — Extreme Stress-Test (max_steps=120, all rules)
# ============================================================================
#
# Scenario
# --------
# The board has called an emergency multi-domain compliance review following
# a regulator's surprise audit letter.  The compliance team must audit 25
# records spanning HR, Finance, Data, and Legal functions — simultaneously
# applying all 10 rules.
#
# Challenges (simultaneous constraint conflicts):
#   1. THREE duplicate-ID groups (ids 5, 12, 20 — each appears twice)
#   2. Records with GDPR + overtime violations; resolving one reveals the other
#   3. Missing fields at varying severity (R10)
#   4. Expired contracts on active employees (R5)
#   5. Salary boundary edge cases (R3)
#   6. Minor employees (R1) — edge cases at age 18 and hours 8
#   7. Dynamic events: POLICY_UPDATE lowers overtime threshold mid-audit;
#      two RECORD_AMENDMENTs resolve violations for specific records;
#      two SYSTEM_OUTAGEs block records temporarily
#
# Pre-event violation summary (35 (record, rule) pairs):
#   RS001: R1          RS002: R2+R3       RS003: R8
#   RS004: R5+R9       RS005: R3+R7       RS006: R4
#   RS007: R6+R8       RS008: R4          RS009: R10
#   RS010: R3+R9       RS011: R7          RS012: R5
#   RS013: R6          RS014: R3          RS015: R8+R9
#   RS016: R4          RS017: R10         RS018: R3+R6
#   RS019: R6          RS020: R2+R8       RS021: —
#   RS022: R4          RS023: R1          RS024: R3+R7+R8
#   RS025: —
#
# After dynamic events fire (varies by seed):
#   RS003:R8 resolved (RECORD_AMENDMENT sets compliance_training=True)
#   RS019:R6 resolved (RECORD_AMENDMENT sets background_check=True)
#   R7 threshold → 40h (POLICY_UPDATE): RS005 and RS024 R7 still hold (hours>40)
#   RS007 and RS015 go into SYSTEM_OUTAGE (agent must wait)
# ============================================================================
_REGULATORY_STORM_RECORDS: List[RecordTruth] = [
    # RS001: Minor employee (age 16) working 10 hours — R1
    RecordTruth(
        record_id="RS001",
        fields={
            "id": 1, "name": "Ada Lewis", "age": 16, "role": "employee",
            "hours": 10, "salary": 35000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R1"],
    ),
    # RS002: Intern (age 20) 44 hours (R2) + salary 38000 > intern max 35000 (R3)
    RecordTruth(
        record_id="RS002",
        fields={
            "id": 2, "name": "Ben Nakamura", "age": 20, "role": "intern",
            "hours": 44, "salary": 38000, "status": "active",
            "contract_end": "2025-09-01", "background_check": False,
            "overtime_approved": False, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R2", "R3"],
    ),
    # RS003: Missing compliance training (R8) — RECORD_AMENDMENT fires later to set True
    RecordTruth(
        record_id="RS003",
        fields={
            "id": 3, "name": "Clara Reich", "age": 33, "role": "analyst",
            "hours": 40, "salary": 60000, "status": "active",
            "contract_end": "2025-06-01", "background_check": False,
            "overtime_approved": True, "compliance_training": False,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R8"],  # resolved by RECORD_AMENDMENT event
    ),
    # RS004: Active contractor, contract expired (R5) + PII access without consent (R9)
    RecordTruth(
        record_id="RS004",
        fields={
            "id": 4, "name": "David Okonkwo", "age": 28, "role": "contractor",
            "hours": 40, "salary": 75000, "status": "active",
            "contract_end": "2023-03-01",  # expired
            "background_check": False, "overtime_approved": True,
            "compliance_training": True, "pii_access": True, "gdpr_consent": False,
        },
        expected_violations=["R5", "R9"],
    ),
    # RS005: Employee 52h, no OT approval (R7) + salary 82000 > employee max 80000 (R3)
    # After POLICY_UPDATE (threshold→40), R7 still applies since 52 > 40
    RecordTruth(
        record_id="RS005",
        fields={
            "id": 5, "name": "Elise Ramos", "age": 31, "role": "employee",
            "hours": 52, "salary": 82000, "status": "active",
            "contract_end": "2025-12-01", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        # R4: shares id=5 with RS006; R3: salary>80000; R7: hours=52>48
        expected_violations=["R3", "R4", "R7"],
    ),
    # RS006: Duplicate ID 5 with RS005 → R4
    RecordTruth(
        record_id="RS006",
        fields={
            "id": 5, "name": "Frank Torres", "age": 29, "role": "employee",
            "hours": 38, "salary": 55000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R4"],
    ),
    # RS007: Manager, no background check (R6) + no compliance training (R8)
    # → SYSTEM_OUTAGE fires for 8 steps; grader accounts for outage window
    RecordTruth(
        record_id="RS007",
        fields={
            "id": 7, "name": "Grace Patel", "age": 38, "role": "manager",
            "hours": 40, "salary": 90000, "status": "active",
            "contract_end": "2025-06-01", "background_check": False,
            "overtime_approved": True, "compliance_training": False,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R6", "R8"],
    ),
    # RS008: Duplicate ID 5 third occurrence? No — ID 12 is the second duplicate group
    # RS008: id=12, duplicate with RS016 → R4
    RecordTruth(
        record_id="RS008",
        fields={
            "id": 12, "name": "Henry Bassett", "age": 44, "role": "director",
            "hours": 42, "salary": 150000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": True, "gdpr_consent": True,
        },
        expected_violations=["R4"],
    ),
    # RS009: Missing salary field entirely → R10 (R3 skips gracefully)
    RecordTruth(
        record_id="RS009",
        fields={
            "id": 9, "name": "Isabel Ferreira", "age": 25, "role": "employee",
            "hours": 40, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },  # salary deliberately omitted
        expected_violations=["R10"],
    ),
    # RS010: Data engineer with PII access + no consent (R9) +
    #         salary 115000 > data_engineer max 110000 (R3)
    RecordTruth(
        record_id="RS010",
        fields={
            "id": 10, "name": "Jorge Almeida", "age": 30, "role": "data_engineer",
            "hours": 40, "salary": 115000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": True, "gdpr_consent": False,
        },
        expected_violations=["R3", "R9"],
    ),
    # RS011: 54 hours, no OT approval (R7); note: after POLICY_UPDATE this still fires
    RecordTruth(
        record_id="RS011",
        fields={
            "id": 11, "name": "Karen Wu", "age": 27, "role": "employee",
            "hours": 54, "salary": 60000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R7"],
    ),
    # RS012: Active with expired contract → R5
    RecordTruth(
        record_id="RS012",
        fields={
            "id": 12, "name": "Liam Osei", "age": 35, "role": "contractor",
            "hours": 38, "salary": 70000, "status": "active",
            "contract_end": "2023-11-01",  # expired
            "background_check": True, "overtime_approved": True,
            "compliance_training": True, "pii_access": False, "gdpr_consent": False,
        },
        # R4: shares id=12 with RS008 and RS016; R5: expired contract
        expected_violations=["R4", "R5"],
    ),
    # RS013: Director, no background check (R6)
    RecordTruth(
        record_id="RS013",
        fields={
            "id": 13, "name": "Maya Singh", "age": 47, "role": "director",
            "hours": 45, "salary": 160000, "status": "active",
            "contract_end": "2025-06-01", "background_check": False,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R6"],
    ),
    # RS014: Finance manager salary 140000 > max 130000 (R3)
    RecordTruth(
        record_id="RS014",
        fields={
            "id": 14, "name": "Nadia Kovac", "age": 42, "role": "finance_manager",
            "hours": 40, "salary": 140000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R3"],
    ),
    # RS015: ML engineer, PII access + no consent (R9) + no training (R8)
    # → SYSTEM_OUTAGE fires for 6 steps
    RecordTruth(
        record_id="RS015",
        fields={
            "id": 15, "name": "Omar Diaz", "age": 31, "role": "ml_engineer",
            "hours": 40, "salary": 100000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": False,
            "pii_access": True, "gdpr_consent": False,
        },
        expected_violations=["R8", "R9"],
    ),
    # RS016: id=12 duplicate group — R4 (with RS008 and RS012)
    RecordTruth(
        record_id="RS016",
        fields={
            "id": 12, "name": "Petra Novak", "age": 29, "role": "analyst",
            "hours": 38, "salary": 68000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R4"],
    ),
    # RS017: Missing role field → R10; salary present but R3 skips gracefully
    RecordTruth(
        record_id="RS017",
        fields={
            "id": 17, "name": "Quinn Huang",
            "hours": 38, "salary": 55000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },  # role deliberately omitted
        expected_violations=["R10"],
    ),
    # RS018: Accountant, no background check (R6) + salary 38000 < accountant min 40000 (R3)
    RecordTruth(
        record_id="RS018",
        fields={
            "id": 18, "name": "Rachel Musa", "age": 26, "role": "accountant",
            "hours": 40, "salary": 38000, "status": "active",
            "contract_end": "2025-06-01", "background_check": False,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R3", "R6"],
    ),
    # RS019: Manager, no background check (R6) — RECORD_AMENDMENT fires to set True
    RecordTruth(
        record_id="RS019",
        fields={
            "id": 19, "name": "Samuel Ito", "age": 34, "role": "manager",
            "hours": 40, "salary": 95000, "status": "active",
            "contract_end": "2025-06-01", "background_check": None,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R6"],  # resolved by RECORD_AMENDMENT event
    ),
    # RS020: Intern (age 22) 43 hours (R2) + no compliance training (R8)
    RecordTruth(
        record_id="RS020",
        fields={
            "id": 20, "name": "Tara Obi", "age": 22, "role": "intern",
            "hours": 43, "salary": 25000, "status": "active",
            "contract_end": "2025-06-01", "background_check": False,
            "overtime_approved": False, "compliance_training": False,
            "pii_access": False, "gdpr_consent": False,
        },
        # R2: intern >40h; R4: shares id=20 with RS022; R8: no compliance training
        expected_violations=["R2", "R4", "R8"],
    ),
    # RS021: Fully compliant employee (edge case: age exactly 18, hours exactly 40)
    RecordTruth(
        record_id="RS021",
        fields={
            "id": 21, "name": "Uma Larsson", "age": 18, "role": "employee",
            "hours": 40, "salary": 35000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=[],  # age=18 not <18; hours=40 not >40
    ),
    # RS022: id=20 duplicate group — R4 (with a hypothetical RS025? No — use id 20 with RS020)
    # RS022 shares id=20 with RS020 → R4
    RecordTruth(
        record_id="RS022",
        fields={
            "id": 20, "name": "Victor Chen", "age": 36, "role": "employee",
            "hours": 38, "salary": 58000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R4"],
    ),
    # RS023: Minor (age 17) working 9 hours → R1
    RecordTruth(
        record_id="RS023",
        fields={
            "id": 23, "name": "Wendy Park", "age": 17, "role": "employee",
            "hours": 9, "salary": 32000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": False, "compliance_training": True,
            "pii_access": False, "gdpr_consent": False,
        },
        expected_violations=["R1"],
    ),
    # RS024: Senior analyst with 46h no OT approval (R7, becomes violation after threshold→40) +
    #         salary 95000 > senior_analyst max 90000 (R3) + no compliance_training (R8)
    # NOTE: Before POLICY_UPDATE, R7 does NOT apply (46 ≤ 48). After POLICY_UPDATE (→40), R7 applies.
    # The expected_violations uses the pre-event state (static ground truth).
    # The environment re-evaluates R7 against live policy_overrides but graders
    # use the event_schedule to determine the correct set at episode end.
    RecordTruth(
        record_id="RS024",
        fields={
            "id": 24, "name": "Xavier Holm", "age": 29, "role": "senior_analyst",
            "hours": 46, "salary": 95000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": False, "compliance_training": False,
            "pii_access": False, "gdpr_consent": False,
        },
        # R3: salary 95000 > senior_analyst max 90000
        # R8: no compliance training
        # R7: hours=46 <= 48 default threshold, so R7 does NOT fire pre-event.
        #     After POLICY_UPDATE (threshold→40), R7 fires. The ExtremeAuditGrader
        #     handles this dynamically via policy_overrides; pre-event ground truth here
        #     only reflects static rule evaluation (no policy override).
        expected_violations=["R3", "R8"],
    ),
    # RS025: Fully compliant manager (all fields present, within band, all checks done)
    RecordTruth(
        record_id="RS025",
        fields={
            "id": 25, "name": "Yuki Brennan", "age": 40, "role": "manager",
            "hours": 38, "salary": 100000, "status": "active",
            "contract_end": "2025-06-01", "background_check": True,
            "overtime_approved": True, "compliance_training": True,
            "pii_access": True, "gdpr_consent": True,
        },
        expected_violations=[],
    ),
]
# Also flag duplicate IDs for RS020/RS022 (id=20) and RS008/RS012/RS016 (id=12)
# and RS005/RS006 (id=5) — the R4 expected_violations above already include these.


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASKS: Dict[str, TaskDefinition] = {
    "easy_basic_audit": TaskDefinition(
        task_id="easy_basic_audit",
        title="Basic HR Compliance Audit",
        difficulty="easy",
        objective=(
            "Audit 5 employee records against 2 compliance rules (R1: minor overhours, "
            "R2: intern overhours). Inspect each record, identify all violations, and generate "
            "a final audit report."
        ),
        max_steps=25,
        active_rule_ids=["R1", "R2"],
        records=_EASY_RECORDS,
    ),
    "medium_mixed_audit": TaskDefinition(
        task_id="medium_mixed_audit",
        title="Mixed HR & Payroll Compliance Audit",
        difficulty="medium",
        objective=(
            "Audit 12 employee records against 4 rules (R1-R4). Some violations are non-obvious: "
            "check salary ranges per role (R3) and duplicate employee IDs (R4). Generate a "
            "complete audit report with no missed violations and no false positives."
        ),
        max_steps=50,
        active_rule_ids=["R1", "R2", "R3", "R4"],
        records=_MEDIUM_RECORDS,
    ),
    "hard_complex_audit": TaskDefinition(
        task_id="hard_complex_audit",
        title="Full Regulatory Compliance Audit",
        difficulty="hard",
        objective=(
            "Audit 20 records against all 5 rules (R1-R5). Handle overlapping violations "
            "(a record may violate multiple rules), edge cases at rule boundaries, expired-contract "
            "checks (R5), and duplicate IDs. Maximise detected violations with zero false positives "
            "while using as few steps as possible."
        ),
        max_steps=100,
        active_rule_ids=["R1", "R2", "R3", "R4", "R5"],
        records=_HARD_RECORDS,
    ),
    "finance_sox_audit": TaskDefinition(
        task_id="finance_sox_audit",
        title="Finance Department SOX Compliance Audit",
        difficulty="hard",
        objective=(
            "Conduct a Sarbanes-Oxley (SOX) pre-certification audit on 15 Finance department "
            "records. Apply 4 rules: R3 (salary within role band), R6 (background check for "
            "sensitive roles — finance_manager, accountant, director), R7 (overtime >48 h/week "
            "requires explicit approval), and R8 (all active employees must have completed annual "
            "compliance training). There are 17 violation pairs across the 15 records — including "
            "overlapping violations on single records and edge cases (e.g., exactly 48 hours is "
            "NOT overtime). Achieve full detection with zero false positives."
        ),
        max_steps=80,
        active_rule_ids=["R3", "R6", "R7", "R8"],
        records=_FINANCE_SOX_RECORDS,
    ),
    "gdpr_privacy_audit": TaskDefinition(
        task_id="gdpr_privacy_audit",
        title="GDPR Data-Privacy Compliance Audit",
        difficulty="medium",
        objective=(
            "Conduct a GDPR data-privacy audit on the Engineering & Analytics team (10 records). "
            "Apply 3 rules: R5 (active employees must have a valid, non-expired contract), "
            "R8 (all active employees must have completed annual data-privacy training), and "
            "R9 (employees with PII data access must have recorded GDPR consent). "
            "Key challenge: inactive employees (status='inactive') are exempt from R5 and R8; "
            "employees without pii_access=True are exempt from R9. Avoid false positives on "
            "these exemptions while finding all 9 true violation pairs."
        ),
        max_steps=50,
        active_rule_ids=["R5", "R8", "R9"],
        records=_GDPR_PRIVACY_RECORDS,
    ),
    "data_integrity_audit": TaskDefinition(
        task_id="data_integrity_audit",
        title="HR Data Integrity Audit",
        difficulty="medium",
        objective=(
            "Audit 8 HR records for data-integrity violations. Apply 3 rules: "
            "R3 (salary within role band), R4 (no duplicate employee IDs), and "
            "R10 (all records must have id, name, role, hours, salary populated). "
            "Some records have fields deliberately omitted — a common real-world data "
            "quality issue that causes payroll errors and compliance failures. "
            "Key challenge: a record missing 'salary' will NOT trigger R3 (rule skips "
            "gracefully), but WILL trigger R10. Find all 6 violation pairs with zero "
            "false positives."
        ),
        max_steps=40,
        active_rule_ids=["R3", "R4", "R10"],
        records=_DATA_INTEGRITY_RECORDS,
    ),
    "regulatory_storm_audit": TaskDefinition(
        task_id="regulatory_storm_audit",
        title="Regulatory Storm: Multi-Domain Stress-Test Audit",
        difficulty="extreme",
        objective=(
            "Extreme stress-test audit across 25 records covering all 10 compliance rules "
            "simultaneously. Three simultaneous constraint conflicts: (1) duplicate-ID groups "
            "across three different ID values, (2) records with overlapping GDPR + overtime "
            "violations where resolving the evidence for one reveals the other, and (3) missing "
            "fields at varying severity levels triggering R10. Mid-episode dynamic events fire: "
            "a POLICY_UPDATE lowers the overtime threshold from 48 to 40 h (requiring re-evaluation "
            "of all records with 41-47 hours), SYSTEM_OUTAGE blocks two records for multiple steps, "
            "and two RECORD_AMENDMENTs correct violations that were valid before the amendment. "
            "An agent that memorises static task data will fail — correct answers depend on the "
            "seed-dependent event schedule. Score ≥ 0.50 requires zero false positives and "
            "≥ 50% detection. Score 1.0 requires full detection + zero false positives + efficient coverage."
        ),
        max_steps=120,
        active_rule_ids=["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
        records=_REGULATORY_STORM_RECORDS,
    ),
}


def list_task_ids() -> List[str]:
    return list(TASKS.keys())
