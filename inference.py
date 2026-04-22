"""Inference script for OpenEnv Compliance Audit.

Mandatory environment variables:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN

This script emits strict evaluator logs in this format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
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
BENCHMARK = os.getenv("BENCHMARK", "openenv_compliance_audit")
# Required by some evaluators when environments are started from docker images.
# Not used by this script because it runs the local environment class directly.
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")
RUN_LOG_PATH = os.getenv("RUN_LOG_PATH", "model-benchmark-logs/inference_runs.jsonl")

# ---------------------------------------------------------------------------
# Inference hyper-parameters
# ---------------------------------------------------------------------------
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 256
FALLBACK_FINISH = {"action_type": "finish", "record_id": None, "rule_id": None}

# ---------------------------------------------------------------------------
# System prompt — covers all 10 rules so the agent reasons correctly
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
  prioritize_rules — [STREAMING TASK ONLY] declare your rule priority strategy BEFORE auditing.
                     This is FREE (zero step cost). Provide rule_priority_order as a list of all
                     active rule IDs in the order you want to audit them. Do this FIRST.
  generate_report  — submit the final audit report (ends episode, best terminal bonus)
  finish           — end the episode without a report (lower terminal reward)

JSON schema:
  {"action_type": "<value>", "record_id": "<id or null>", "rule_id": "<id or null>"}
  For prioritize_rules: {"action_type": "prioritize_rules", "rule_priority_order": ["R8", "R3", ...]}

──────────────────────────────────────────────────────────
COMPLIANCE RULES REFERENCE
──────────────────────────────────────────────────────────
R1  Minor overhours          — age < 18 AND hours > threshold (default 8, may change via POLICY_UPDATE)
R2  Intern overhours         — role == 'intern' AND hours > threshold (default 40)
R3  Salary out of range      — salary < role_min OR salary > role_max
                               (each role has its own band; check available_rules for detail)
R4  Duplicate employee ID    — same 'id' value in more than one record
R5  Expired contract active  — contract_end < '2024-01-01' AND status == 'active'
R6  Background check missing — sensitive role (manager/director/finance_manager/accountant/
                               cfo/security/hr/compliance_officer/legal_counsel/vp_finance)
                               AND background_check != True
R7  Unapproved overtime      — hours > threshold (default 48, STRICT >) AND overtime_approved != True
                               ⚠ Exactly threshold hours is NOT a violation (edge case)
                               ⚠ POLICY_UPDATE can lower the threshold mid-episode
R8  Missing compliance train — status == 'active' AND compliance_training != True
                               ⚠ Inactive employees are EXEMPT from this rule
R9  GDPR consent missing     — pii_access == True AND gdpr_consent != True
                               ⚠ If pii_access is False/absent, rule does NOT apply
R10 Missing required fields  — any of {id,name,role,hours,salary} is missing or null
                               ⚠ Zero values are valid; only missing/null is a violation

Only rules listed in 'available_rules' are active for the current task.

──────────────────────────────────────────────────────────
DYNAMIC EVENTS
──────────────────────────────────────────────────────────
Mid-episode events may change the audit landscape:

  POLICY_UPDATE      — A rule's threshold may change (e.g. overtime threshold 48→40).
                       Check 'current_policy_overrides' in the observation EVERY step.
                       Re-evaluate any previously applied rules if the threshold changed.

  SYSTEM_OUTAGE      — A record becomes temporarily inaccessible.
                       'system_outage': true in the record view. Inspecting it returns an error.
                       Wait until the outage ends (visible in the record view) then retry.

  RECORD_AMENDMENT   — A field value is corrected mid-episode.
                       If a record's field changes and its violation is resolved,
                       do NOT flag that violation (it would be a false positive).

Fired events are listed in 'active_events' in the observation.

──────────────────────────────────────────────────────────
OPTIMAL STRATEGY
──────────────────────────────────────────────────────────
1. Inspect every record (inspect_record) — fields are hidden until inspected.
   Skip records in SYSTEM_OUTAGE; come back after outage_ends_at step.
2. For each inspected record, apply_rule for each active rule.
   A positive reward (+0.2) signals a real violation — take note.
   Check current_policy_overrides BEFORE applying R1, R2, R7.
3. For every confirmed violation, call flag_violation.
   Do NOT flag violations on records that had RECORD_AMENDMENT resolving the issue.
4. For records with no violations, call mark_compliant.
5. Call generate_report with a structured payload including audit_confidence section.

──────────────────────────────────────────────────────────
CRITICAL RULES
──────────────────────────────────────────────────────────
• Never repeat the same action twice on the same record (penalty -0.05 / -0.10).
• More than 3 identical (action_type, record_id) in 5 steps = loop penalty (-0.10).
• flag_violation on a compliant record costs -0.3 — be precise.
• Respect rule exemptions: inactive employees skip R5 and R8;
  employees without pii_access skip R9; non-sensitive roles skip R6.
• record_id and rule_id must match exactly what appears in the observation.
• generate_report payload should include audit_confidence section for bonus points.
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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _fmt_action(action: AuditAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "record_id": action.record_id,
        "rule_id": action.rule_id,
    }
    if action.report is not None:
        payload["report"] = action.report
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


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
    model_name: str,
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
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED + step,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return _fallback_action(obs_dict)

    try:
        payload = _extract_json(response_text)
    except Exception:
        payload = _fallback_action(obs_dict)

    payload.setdefault("record_id", None)
    payload.setdefault("rule_id", None)
    return payload


def _append_run_log(entry: Dict[str, Any]) -> None:
    path = Path(RUN_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _action_signature(action: AuditAction) -> tuple[str, Optional[str], Optional[str]]:
    return (action.action_type.value, action.record_id, action.rule_id)


def run_task(client: OpenAI, task_id: str, model_name: str, seed: int = 42) -> tuple[float, str]:
    """Run one full episode and return (final_score, failure_mode)."""
    env = ComplianceAuditEnv(task_id=task_id)
    obs = env.reset(seed=seed).model_dump()
    done = False
    step = 0
    history: List[str] = []
    score = 0.0
    rewards: List[float] = []
    success = False
    prev_sig: Optional[tuple[str, Optional[str], Optional[str]]] = None
    repeat_count = 0
    failure_mode = "none"

    print(f"[START] task={task_id} env={BENCHMARK} model={model_name} seed={seed}")

    try:
        while not done:
            step += 1
            payload = _choose_action(client, obs, history, step, model_name)

            try:
                action = AuditAction.model_validate(payload)
            except Exception:
                action = AuditAction.model_validate(FALLBACK_FINISH)

            # Break out of loops when the model repeats the exact same action.
            sig = _action_signature(action)
            if sig == prev_sig:
                repeat_count += 1
            else:
                repeat_count = 0

            if repeat_count >= 2:
                fallback_payload = _fallback_action(obs)
                action = AuditAction.model_validate(fallback_payload)
                sig = _action_signature(action)
                repeat_count = 0

            prev_sig = sig

            result = env.step(action)
            obs = result.observation.model_dump()
            done = result.done
            reward = float(result.reward)
            rewards.append(reward)
            score = max(0.0, min(1.0, float(result.info.get("task_score", 0.0))))
            failure_mode = str(result.info.get("failure_mode", "none"))
            error_value = obs.get("last_action_error")
            error_text = str(error_value) if error_value else "null"

            action_str = _fmt_action(action)
            history.append(f"step={step} {action_str} reward={reward:+.2f}")
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={_fmt_reward(reward)} done={_bool_text(done)} error={error_text}"
            )

            if done:
                success = score >= 0.1
                break

    except Exception as exc:
        print(
            f"[STEP] step={step} action=null reward={_fmt_reward(0.0)} "
            f"done=true error={str(exc)}"
        )
        success = False
    finally:
        env.close()

    rewards_text = ",".join(_fmt_reward(r) for r in rewards)
    print(
        f"[END] success={_bool_text(success)} steps={step} "
        f"score={score:.2f} rewards={rewards_text} failure_mode={failure_mode}"
    )

    _append_run_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": BENCHMARK,
            "task_id": task_id,
            "model": model_name,
            "seed": seed,
            "score": round(score, 4),
            "steps": step,
            "success": success,
            "failure_mode": failure_mode,
            "rewards": [round(item, 4) for item in rewards],
        }
    )

    return score, failure_mode






def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenEnv Compliance Audit inference")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Task IDs to run (default: all tasks)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "One or more model names for comparison mode. "
            "If omitted, uses MODEL_NAME from environment."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Number of seeds to run per task for variance reporting. "
            "Seeds will be 42, 43, ... 42+(N-1). Default: 1."
        ),
    )
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    models = args.models if args.models else [MODEL_NAME]
    num_seeds = max(1, args.seeds)
    seeds = [42 + i for i in range(num_seeds)]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}  |  SEEDS: {seeds}")
        print(f"{'='*60}")

        # Dict: task_id → list of (score, failure_mode)
        task_results: Dict[str, List[tuple[float, str]]] = {tid: [] for tid in args.tasks}

        for seed in seeds:
            for task_id in args.tasks:
                score, failure_mode = run_task(client, task_id, model_name, seed=seed)
                task_results[task_id].append((score, failure_mode))

        # --- Variance report ---
        print(f"\n{'─'*60}")
        print(f"VARIANCE REPORT — {model_name}")
        print(f"{'─'*60}")
        print(f"{'Task':<30} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}  Failure Modes")
        print(f"{'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*25}")

        all_means: List[float] = []
        for task_id, results in task_results.items():
            scores = [r[0] for r in results]
            modes = [r[1] for r in results]
            mean_s = statistics.mean(scores)
            std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
            min_s = min(scores)
            max_s = max(scores)
            mode_summary = ", ".join(sorted(set(modes)))
            print(
                f"{task_id:<30} {mean_s:>6.3f} {std_s:>6.3f} {min_s:>6.3f} {max_s:>6.3f}  {mode_summary}"
            )
            all_means.append(mean_s)

        overall_mean = statistics.mean(all_means)
        print(f"{'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        print(f"{'OVERALL':<30} {overall_mean:>6.3f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
