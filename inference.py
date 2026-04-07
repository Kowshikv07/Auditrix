"""
Inference Script — OpenEnv Compliance Audit Environment
========================================================
Mandatory environment variables:
    API_BASE_URL   LLM endpoint base URL (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key

Run:
    python inference.py [--tasks easy_basic_audit medium_mixed_audit ...]

The script runs all 5 tasks (easy → medium → hard → finance_sox → gdpr_privacy),
prints per-step details, and prints a final score table.
Expected runtime < 30 min on 2 vCPU / 8 GB RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from openenv_compliance_audit.environment import ComplianceAuditEnv
from openenv_compliance_audit.models import AuditAction
from openenv_compliance_audit.tasks import TASKS

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = (
    os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
)
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ---------------------------------------------------------------------------
# Inference hyper-parameters
# ---------------------------------------------------------------------------
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 256
FALLBACK_FINISH = {"action_type": "finish", "record_id": None, "rule_id": None}

# ---------------------------------------------------------------------------
# System prompt — covers all 9 rules so the agent reasons correctly
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an AI compliance auditor agent operating inside an OpenEnv environment.

On every turn you receive a JSON observation and must output EXACTLY ONE JSON action.
No prose. No markdown fences. Just raw JSON.

──────────────────────────────────────────────────────────
AVAILABLE ACTIONS
──────────────────────────────────────────────────────────
  inspect_record   — reveal a record's fields (REQUIRED before any audit action)
  apply_rule       — run a compliance rule against an inspected record (+0.2 if violation found)
  flag_violation   — officially flag a (record, rule) pair as a violation (+0.5 correct / -0.3 wrong)
  mark_compliant   — declare a record has no violations (+0.05 if truly compliant)
  generate_report  — submit the final audit report (ends episode, best terminal bonus)
  finish           — end the episode without a report (lower terminal reward)

JSON schema:
  {"action_type": "<value>", "record_id": "<id or null>", "rule_id": "<id or null>"}

──────────────────────────────────────────────────────────
COMPLIANCE RULES REFERENCE
──────────────────────────────────────────────────────────
R1  Minor overhours          — age < 18 AND hours > 8
R2  Intern overhours         — role == 'intern' AND hours > 40
R3  Salary out of range      — salary < role_min OR salary > role_max
                               (each role has its own band; check available_rules for detail)
R4  Duplicate employee ID    — same 'id' value in more than one record
R5  Expired contract active  — contract_end < '2024-01-01' AND status == 'active'
R6  Background check missing — sensitive role (manager/director/finance_manager/accountant/
                               cfo/security/hr) AND background_check != True
R7  Unapproved overtime      — hours > 48 (STRICT >) AND overtime_approved != True
                               ⚠ Exactly 48 hours is NOT a violation (edge case)
R8  Missing compliance train — status == 'active' AND compliance_training != True
                               ⚠ Inactive employees are EXEMPT from this rule
R9  GDPR consent missing     — pii_access == True AND gdpr_consent != True
                               ⚠ If pii_access is False/absent, rule does NOT apply

Only rules listed in 'available_rules' are active for the current task.

──────────────────────────────────────────────────────────
OPTIMAL STRATEGY
──────────────────────────────────────────────────────────
1. Inspect every record (inspect_record) — fields are hidden until inspected.
2. For each inspected record, apply_rule for each active rule.
   A positive reward (+0.2) signals a real violation — take note.
3. For every confirmed violation, call flag_violation.
4. For records with no violations, call mark_compliant.
5. Call generate_report when all records are processed.

──────────────────────────────────────────────────────────
CRITICAL RULES
──────────────────────────────────────────────────────────
• Never repeat the same action twice (penalty -0.05).
• flag_violation on a compliant record costs -0.3 — be precise.
• Respect rule exemptions: inactive employees skip R5 and R8;
  employees without pii_access skip R9; non-sensitive roles skip R6.
• record_id and rule_id must match exactly what appears in the observation.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _fallback_action(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic fallback when the model produces unparseable output.

    Priority order:
      1. Inspect any un-inspected record.
      2. Apply any unapplied active rule on each inspected record.
      3. Flag confirmed violations not yet flagged.
      4. Mark records with all rules applied and no violations as compliant.
      5. generate_report.
    """
    records = obs_dict.get("visible_records", [])
    active_rules = [r["rule_id"] for r in obs_dict.get("available_rules", [])]
    flagged_pairs = {
        (v["record_id"], v["rule_id"]) for v in obs_dict.get("violations_found", [])
    }

    # Step 1: inspect un-inspected records
    for rec in records:
        if not rec.get("inspected"):
            return {"action_type": "inspect_record", "record_id": rec["record_id"], "rule_id": None}

    # Step 2: apply unapplied rules on inspected records
    for rec in records:
        if not rec.get("inspected"):
            continue
        rec_flags = set(rec.get("flags", []))
        for rule_id in active_rules:
            pair = (rec["record_id"], rule_id)
            if rule_id not in rec_flags and pair not in flagged_pairs:
                # Use apply_rule to probe — the environment will tell us if it's a violation
                return {
                    "action_type": "apply_rule",
                    "record_id": rec["record_id"],
                    "rule_id": rule_id,
                }

    # Step 3: flag any violations already discovered but not yet flagged
    for v in obs_dict.get("violations_found", []):
        return {
            "action_type": "flag_violation",
            "record_id": v["record_id"],
            "rule_id": v["rule_id"],
        }

    return {"action_type": "generate_report", "record_id": None, "rule_id": None}


def _choose_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    history: List[str],
    step: int,
) -> Dict[str, Any]:
    """Ask the LLM for the next action."""
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "step": step,
                    "remaining_steps": obs_dict.get("remaining_steps"),
                    "objective": obs_dict.get("objective"),
                    "available_rules": obs_dict.get("available_rules", []),
                    "visible_records": obs_dict.get("visible_records", []),
                    "checked_records": obs_dict.get("checked_records", []),
                    "violations_found": obs_dict.get("violations_found", []),
                    "recent_actions": history[-8:],
                    "last_action_error": obs_dict.get("last_action_error"),
                },
                indent=None,
                separators=(",", ":"),
            ),
        }
    ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED + step,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Model request failed ({exc}). Using fallback action.")
        return _fallback_action(obs_dict)

    try:
        payload = _extract_json(response_text)
    except Exception:
        payload = _fallback_action(obs_dict)

    payload.setdefault("record_id", None)
    payload.setdefault("rule_id", None)
    return payload


def run_task(client: OpenAI, task_id: str) -> float:
    """Run one full episode and return the final task score."""
    env = ComplianceAuditEnv(task_id=task_id)
    obs = env.reset().model_dump()
    done = False
    step = 0
    history: List[str] = []
    score = 0.0

    task_meta = TASKS[task_id]
    total_violations = sum(len(r.expected_violations) for r in task_meta.records)
    print(
        f"\n{'='*65}\n[TASK] {task_id}\n"
        f"       Difficulty : {task_meta.difficulty.upper()}\n"
        f"       Records    : {len(task_meta.records)}\n"
        f"       Rules      : {task_meta.active_rule_ids}\n"
        f"       Violations : {total_violations} (ground truth)\n"
        f"       Max steps  : {task_meta.max_steps}\n{'='*65}"
    )

    try:
        while not done:
            step += 1
            payload = _choose_action(client, obs, history, step)

            try:
                action = AuditAction.model_validate(payload)
            except Exception:
                action = AuditAction.model_validate(FALLBACK_FINISH)

            result = env.step(action)
            obs = result.observation.model_dump()
            done = result.done
            reward = float(result.reward)
            score = float(result.info.get("task_score", 0.0))
            error = obs.get("last_action_error") or "—"

            action_str = json.dumps(
                {"t": action.action_type.value, "rec": action.record_id, "rule": action.rule_id},
                separators=(",", ":"),
            )
            history.append(f"step={step} {action_str} reward={reward:+.2f}")

            print(
                f"  Step {step:3d}: {action_str:<58} "
                f"reward={reward:+.2f}  score={score:.3f}  err={error}"
            )

            if result.done:
                print("  → Episode complete.")
                break

            if step >= task_meta.max_steps:
                print(f"  → Max steps ({task_meta.max_steps}) reached.")
                break

    except Exception as exc:
        print(f"  [ERROR] {exc}")
    finally:
        env.close()

    print(f"[RESULT] task={task_id}  final_score={score:.4f}  steps_used={step}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEnv Compliance Audit inference")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Task IDs to run (default: all 5)",
    )
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores: Dict[str, float] = {}
    for task_id in args.tasks:
        scores[task_id] = run_task(client, task_id)

    avg = sum(scores.values()) / len(scores) if scores else 0.0

    print(f"\n{'='*65}")
    print("FINAL SCORES")
    print(f"{'='*65}")
    for tid, sc in scores.items():
        diff = TASKS[tid].difficulty.upper()
        print(f"  {tid:<38} [{diff:>6}]  {sc:.4f}")
    print(f"  {'AVERAGE':<38}          {avg:.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
