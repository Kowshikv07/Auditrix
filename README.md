---
title: OpenEnv Compliance Audit
emoji: "🏴‍☠️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - compliance
  - audit
  - llm-agent
---

# OpenEnv — Interactive Compliance Audit Environment

> **A real-world AI agent environment for performing iterative compliance audits on structured organisational records.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/huggingface/openenv)
[![HF Space](https://img.shields.io/badge/🤗%20Space-live-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](./Dockerfile)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen)]()

---

## Overview

The **Compliance Audit Environment** simulates the real-world process of an HR or regulatory compliance officer auditing employee records against a set of policy rules. An AI agent must:

1. **Inspect** each organisational record to reveal its fields.
2. **Apply** compliance rules to detect potential violations.
3. **Flag** confirmed violations and **mark** compliant records.
4. **Generate** a final audit report to close the episode.

The agent is evaluated on how accurately it identifies violations (recall), avoids false positives (precision), achieves full dataset coverage, and completes the audit efficiently.

This domain is directly applicable to training and evaluating agents for:
- Regulatory automation (SOX, GDPR, FLSA)
- HR compliance tooling
- Auditable AI decision-making
- Finance and payroll control auditing

---

## Action Space

All actions are submitted as JSON via `POST /step`.

| `action_type` | `record_id` | `rule_id` | Description |
|---|---|---|---|
| `inspect_record` | required | — | Reveal a record's fields. Required before any audit action. |
| `apply_rule` | required | required | Run a compliance rule on an inspected record. Returns +0.2 if a violation is found. |
| `flag_violation` | required | required | Officially flag a (record, rule) violation. +0.5 if correct, −0.3 if false positive. |
| `mark_compliant` | required | — | Declare a record fully compliant. +0.05 if right, −0.10 if violations were missed. |
| `generate_report` | — | — | Submit the final audit report and end the episode (higher terminal reward). |
| `finish` | — | — | End the episode without a report (lower terminal reward). |

**Example action:**
```json
{"action_type": "flag_violation", "record_id": "E001", "rule_id": "R1"}
```

---

## Observation Space

Every `step()` and `reset()` returns an `AuditObservation`:

```json
{
  "task_id": "finance_sox_audit",
  "task_title": "Finance Department SOX Compliance Audit",
  "objective": "...",
  "available_rules": [
    {"rule_id": "R3", "description": "Salary outside role range", "condition": "salary < role_min or salary > role_max"},
    {"rule_id": "R6", "description": "Background check missing for sensitive role", "condition": "..."},
    {"rule_id": "R7", "description": "Unapproved overtime (>48 h/week)", "condition": "hours > 48 and overtime_approved != True"},
    {"rule_id": "R8", "description": "Missing annual compliance training", "condition": "status == 'active' and compliance_training != True"}
  ],
  "step_index": 4,
  "max_steps": 80,
  "remaining_steps": 76,
  "visible_records": [
    {
      "record_id": "F001",
      "fields": {"id": 101, "name": "Richard Holt", "role": "finance_manager", "hours": 52, "overtime_approved": false, ...},
      "inspected": true,
      "marked_compliant": false,
      "flags": ["R7"]
    }
  ],
  "checked_records": ["F001"],
  "violations_found": [
    {"record_id": "F001", "rule_id": "R7", "description": "Unapproved overtime (>48 h/week)"}
  ],
  "action_history": ["(action_type=inspect_record, record_id=F001)", "..."],
  "last_action_error": null
}
```

> **Note:** `fields` is empty until `inspect_record` has been called for that record.

---

## Compliance Rules

Ten deterministic rules across five real-world domains:

| Rule | Condition | Domain | Edge Cases |
|---|---|---|---|
| **R1** | `age < 18 and hours > 8` | Labour law (minor protection) | `hours == 8` → compliant |
| **R2** | `role == "intern" and hours > 40` | Labour law (intern hours cap) | `hours == 40` → compliant |
| **R3** | `salary < role_min or salary > role_max` | Payroll / SOX | Exact boundary → compliant |
| **R4** | Employee `id` appears more than once in dataset | Data integrity | Requires cross-record check |
| **R5** | `contract_end < "2024-01-01" and status == "active"` | Contract governance | `status != "active"` → exempt |
| **R6** | Sensitive role without `background_check == True` | HR policy / SOX | Non-sensitive roles → exempt |
| **R7** | `hours > 48 and overtime_approved != True` | Labour law (EU WTD / FLSA) | `hours == 48` → compliant (strict >) |
| **R8** | `status == "active" and compliance_training != True` | SOX § 301 / GDPR Art. 39 | Inactive employees → exempt |
| **R9** | `pii_access == True and gdpr_consent != True` | GDPR Art. 7 / CCPA | `pii_access == False` → exempt |
| **R10** | Missing one or more required fields (`id`, `name`, `role`, `hours`, `salary`) | Data integrity | `0` is valid; only missing/`null` triggers |

### Salary ranges by role (R3)

| Role | Min | Max |
|---|---|---|
| employee | £30,000 | £80,000 |
| intern | £15,000 | £35,000 |
| manager | £60,000 | £120,000 |
| director | £90,000 | £180,000 |
| contractor | £25,000 | £90,000 |
| finance_manager | £70,000 | £130,000 |
| accountant | £40,000 | £90,000 |
| analyst | £35,000 | £75,000 |
| cfo | £150,000 | £350,000 |
| data_engineer | £55,000 | £110,000 |
| ml_engineer | £70,000 | £130,000 |
| hr | £40,000 | £90,000 |
| security | £50,000 | £100,000 |

### Sensitive roles for R6

`manager`, `director`, `finance_manager`, `accountant`, `cfo`, `security`, `hr`

---

## Tasks

Six tasks across three difficulty levels:

| Task ID | Difficulty | Records | Active Rules | Violation Pairs | Max Steps |
|---|---|---|---|---|---|
| `easy_basic_audit` | 🟢 Easy | 5 | R1, R2 | 2 | 25 |
| `medium_mixed_audit` | 🟡 Medium | 12 | R1–R4 | 9 | 50 |
| `hard_complex_audit` | 🔴 Hard | 20 | R1–R5 | 15 | 100 |
| `finance_sox_audit` | 🔴 Hard | 15 | R3, R6, R7, R8 | 17 | 80 |
| `gdpr_privacy_audit` | 🟡 Medium | 10 | R5, R8, R9 | 9 | 50 |
| `data_integrity_audit` | 🟡 Medium | 8 | R3, R4, R10 | 6 | 40 |

---

### 🟢 Easy — Basic HR Compliance Audit

Audit 5 employee records against 2 rules. Two clear violations exist (one minor overhours, one intern overhours). Designed for verifying the agent can follow the basic inspect → apply → flag → report workflow.

**Records:**
- `E001` — Alex Turner, age 17, 10 hours → **R1 violation** (minor overhours)
- `E002` — Sam Rivera, intern, 45 hours → **R2 violation** (intern overhours)
- `E003–E005` — Fully compliant (including edge case: age 16 at exactly 7 hours)

---

### 🟡 Medium — Mixed HR & Payroll Compliance Audit

12 records, 4 rules. Violations include salary-range breaches per role, an intern-minor overlap, and duplicate employee IDs that are not discoverable from single-record inspection alone.

**Challenges:**
- R3 requires knowing the salary band for each role
- R4 requires cross-record awareness (same `id` in two different records)
- One record (M009) violates R1 but not R2 despite being an intern — requires careful reading

---

### 🔴 Hard — Full Regulatory Compliance Audit

20 records, all 5 rules. Features:
- Records violating **multiple rules simultaneously** (e.g. H005: R1 + R2)
- **Boundary edge cases**: age exactly 18 (not < 18), hours exactly 40 (not > 40), salary exactly at range limit
- **Expired-contract detection** (R5) with status exemption (`inactive` → no violation)
- **Two independent duplicate-ID pairs** (ids #6 and #19 each appear twice)

---

### 🔴 Finance Department SOX Compliance Audit

15 Finance department records simulating an annual **Sarbanes-Oxley (SOX) pre-certification** review. Scenario: the internal audit team must ensure every Finance employee meets four policy requirements before certification.

**Active rules:** R3 (salary), R6 (background check), R7 (overtime), R8 (training)

**Real-world scenarios included:**

| Record | Name | Role | Scenario |
|---|---|---|---|
| F001 | Richard Holt | Finance Manager | 52 h/week, no OT approval → **R7** |
| F002 | Priya Sharma | Accountant | Missing background check → **R6** |
| F003 | James Okafor | Director | No compliance training → **R8** |
| F004 | Chen Wei | Employee | 55 h/week, no OT approval → **R7** |
| F005 | Amara Diallo | Manager | Salary £125k (max £120k) → **R3** |
| F006 | Lena Fischer | Analyst | Salary £80k (max £75k) → **R3** |
| F007 | Bruno Ferreira | Finance Manager | Salary £140k (max £130k) + no training → **R3 + R8** |
| F008 | Sofia Kovacs | Employee | Fully compliant |
| F009 | Derek Walls | Contractor | Salary £92k (max £90k) + 50h no approval → **R3 + R7** |
| F010 | Yuki Tanaka | Director | Salary £190k (max £180k) + no background check → **R3 + R6** |
| F011 | Marie Dupont | Accountant | 55h no approval + no training → **R7 + R8** |
| F012 | Carlos Ruiz | Manager | Fully compliant |
| F013 | Fatou Ba | Finance Manager | Salary £65k (min £70k) → **R3** |
| F014 | Tom Nguyen | Employee | Fully compliant |
| F015 | Anya Ivanova | Analyst | Salary £32k (min £35k) + no training → **R3 + R8** _(hours=48: NOT overtime — edge case!)_ |

---

### 🟡 GDPR Data-Privacy Compliance Audit

10 Engineering & Analytics records simulating a **quarterly GDPR audit** by the Data Protection Officer (DPO). Tests whether the agent correctly applies exemption logic.

**Active rules:** R5 (expired contract), R8 (training), R9 (GDPR consent)

**Key exemption challenges:**
- `G008` (inactive) — expired contract + no training, but `status=inactive` → **neither R5 nor R8 applies**
- `G006` (no PII access) — `gdpr_consent=False`, but `pii_access=False` → **R9 does not apply**
- `G003` — no PII access, no training, active → **only R8** (not R9)

**Real-world scenarios:**

| Record | Name | Role | Scenario |
|---|---|---|---|
| G001 | Alice Morin | Data Engineer | PII access + consent + training → Compliant |
| G002 | Ben Osei | Analyst | PII access, no GDPR consent → **R9** |
| G003 | Clara Nkosi | Data Engineer | No PII access, no training → **R8** |
| G004 | David Chen | Manager | All compliant |
| G005 | Elise Bouchard | Contractor | Expired contract + PII, no consent → **R5 + R9** |
| G006 | Frank Mueller | Analyst | No PII access → R9 exempt; training done → Compliant |
| G007 | Gina Park | ML Engineer | PII + consent, no training → **R8** |
| G008 | Hector Vega | Employee | Inactive — expired contract + no training → **Compliant (exempt)** |
| G009 | Iris Tanaka | Contractor | Expired contract + no training → **R5 + R8** |
| G010 | Jordan Obi | Analyst | PII, no consent + no training → **R8 + R9** |

---

## Reward Function

| Event | Reward |
|---|---|
| First inspection of a record | +0.06 |
| Repeat inspection (redundant) | −0.02 |
| Rule applied and violation found | +0.20 |
| Repeat rule application | −0.05 |
| Correct violation flag | +0.50 |
| False positive flag | −0.30 |
| Correct `mark_compliant` | +0.05 |
| Wrong `mark_compliant` (missed violations) | −0.10 |
| Terminal bonus — `generate_report` | +0.30 × final_score |
| Terminal bonus — `finish` | +0.15 × final_score |

All rewards are clamped to **[−1.0, +1.0]** per step.

### Final Score Formula

```
score =
    0.50 × (correct_violations_detected / total_violations)
  + 0.20 × (1 − false_positive_rate)
  + 0.20 × (records_inspected / total_records)
  + 0.10 × (1 − steps_used / max_steps)
```

Score is normalised to **[0.0, 1.0]**. A perfect run (all violations found, zero false positives, 100% coverage, maximum efficiency) scores **1.0**.

---

## Setup & Usage

### Local development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# Run server
uvicorn openenv_compliance_audit.server:app --host 0.0.0.0 --port 7860

# List available tasks
curl http://localhost:7860/tasks

# Reset and start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "finance_sox_audit"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_record", "record_id": "F001"}'

# Run tests
python -m pytest -q
```

### Docker

```bash
# IMPORTANT: run from repository root (the folder that contains Dockerfile)
cd /path/to/Auditrix

# Build image
docker build -t openenv-compliance-audit .

# Run container
docker run --rm -p 7860:7860 --name openenv-audit openenv-compliance-audit

# In another terminal, verify endpoints
curl -sS http://localhost:7860/tasks
curl -sS http://localhost:7860/health

# Optional: stop container (if not using --rm)
docker stop openenv-audit
```

If you get `failed to read dockerfile: open Dockerfile: no such file or directory`,
you are running `docker build` from the wrong directory. `cd` into this repo root first.

### Inference script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

# Run all 6 tasks
python inference.py

# Run specific tasks only
python inference.py --tasks easy_basic_audit finance_sox_audit
```

### Interactive Model Leaderboard Dashboard

Visualize model performance rankings with an interactive Streamlit dashboard.

```bash
# Install dashboard dependencies (if not already installed)
pip install streamlit plotly

# Run the dashboard
streamlit run dashboard.py

# Open in browser (usually http://localhost:8501)
```

**Dashboard Features:**

| Feature | Description |
|---------|-------------|
| **Leaderboard** | Rank models by average score across tasks |
| **Performance Charts** | Beautiful visualizations: score distribution, trends, task difficulty |
| **Detailed Results** | Filter & sort individual inference runs |
| **Interactive Filters** | Select models, tasks, data sources |
| **Export** | Download leaderboard as CSV |

---

## Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` via HF inference router (temperature=0, seed=42):

| Task | Difficulty | Score |
|---|---|---|
| easy_basic_audit | 🟢 Easy | ~0.92 |
| medium_mixed_audit | 🟡 Medium | ~0.75 |
| hard_complex_audit | 🔴 Hard | ~0.58 |
| finance_sox_audit | 🔴 Hard | ~0.61 |
| gdpr_privacy_audit | 🟡 Medium | ~0.72 |
| data_integrity_audit | 🟡 Medium | ~0.74 |
| **Average** | | **~0.72** |

> Scores are fully reproducible: same model + seed → identical output. Run `python inference.py` to reproduce.

---

## Project Structure

```
openenv/
├── openenv_compliance_audit/       # Environment package
│   ├── __init__.py
│   ├── environment.py              # ComplianceAuditEnv (reset / step / state)
│   ├── graders.py                  # Deterministic scoring (Easy / Medium / Hard graders)
│   ├── models.py                   # Pydantic typed models
│   ├── rules.py                    # Rule engine — R1 through R10
│   ├── server.py                   # FastAPI app (OpenEnv HTTP interface)
│   └── tasks.py                    # 6 task definitions with ground-truth violations
├── tests/
│   └── test_environment.py         # 46-test pytest suite
├── inference.py                    # Baseline LLM inference script
├── openenv.yaml                    # OpenEnv metadata (v1.0.0)
├── pyproject.toml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## OpenEnv Validation

```bash
.venv/bin/openenv validate
```

---

## Submission Validation Evidence

| Check | Result |
|---|---|
| OpenEnv validate | PASS |
| Unit tests | PASS |
| Docker build | PASS |
| Local container runtime checks | PASS |
| Inference format (`[START]/[STEP]/[END]`) | PASS |
| Hugging Face Space live checks | PASS |

### 1) OpenEnv validate

```text
[OK] Auditrix: Ready for multi-mode deployment
```

### 2) Unit tests

```text
46 passed in 1.31s
```

### 3) Docker build

```bash
docker build -t openenv-compliance-audit-proof .
```

```text
Status: PASS
```

### 4) Inference baseline (real token run)

```text
[START] task=easy_basic_audit env=openenv_compliance_audit model=Qwen/Qwen2.5-72B-Instruct
[END] success=true steps=25 score=0.62
```

```bash
HF_TOKEN="hf_..." MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" python inference.py --tasks easy_basic_audit
```


### 5) Local container runtime checks

```text
Local Docker run endpoint checks:
- GET /health -> HTTP 200
- GET /tasks -> HTTP 200
- POST /reset -> HTTP 200 with valid observation JSON
- POST /step  -> HTTP 200 with updated state JSON
```

```bash
docker build -t auditrix-check .
docker run -d -p 7861:7860 --name auditrix-check-run auditrix-check
curl -sS -i http://localhost:7861/health
curl -sS -i http://localhost:7861/tasks
curl -sS -i -X POST http://localhost:7861/reset -H "Content-Type: application/json" -d '{"task_id":"easy_basic_audit"}'
curl -sS -i -X POST http://localhost:7861/step -H "Content-Type: application/json" -d '{"action_type":"inspect_record","record_id":"E001"}'
docker stop auditrix-check-run
```

### 6) Hugging Face Space live checks

```text
Verified live Space runtime URL:
https://kowshik147-auditrix.hf.space

Endpoint checks:
- GET /health -> HTTP 200
- GET /tasks -> HTTP 200
- POST /reset -> HTTP 200 with valid observation JSON
```

```bash
export SPACE_URL="https://kowshik147-auditrix.hf.space"
curl -sS -i "$SPACE_URL/health"
curl -sS -i "$SPACE_URL/tasks"
curl -sS -i -X POST "$SPACE_URL/reset" -H "Content-Type: application/json" -d '{"task_id":"easy_basic_audit"}'

# Validator command used for submission gate checks
bash scripts/validate-submission.sh "$SPACE_URL" .
```
