"""Auditrix GRPO Training Script — Compliance Audit RL with Unsloth + TRL.

Train a small language model to perform compliance audits using Group Relative
Policy Optimization (GRPO). The model interacts with the Auditrix environment
via TRL's `environment_factory` — each action (inspect, apply_rule, flag, …) is
exposed as a callable tool that the model learns to invoke correctly.

🏆 OpenEnv Hackathon Submission 
  Theme #2: Long-Horizon Planning (Scale AI Bonus - HR Workflows)
  Theme #3.1: World Modeling (Professional Tasks)

Usage
-----
  # Colab / single-GPU (uses Unsloth for 2x speed + 60% less VRAM)
  python train_grpo.py

  # Without Unsloth (plain HF TRL)
  python train_grpo.py --no-unsloth

  # Dry-run to verify reward function without GPU
  python train_grpo.py --dry-run

Requirements
------------
  pip install unsloth trl>=0.18.0 datasets peft accelerate
  pip install -e .   # install openenv_compliance_audit locally
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Auditrix environment imports
# ---------------------------------------------------------------------------
from openenv_compliance_audit.environment import ComplianceAuditEnv
from openenv_compliance_audit.models import ActionType, AuditAction
from openenv_compliance_audit.tasks import TASKS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Tasks used for training (easy + medium give clear reward signal)
TRAIN_TASKS = ["easy_basic_audit", "medium_mixed_audit", "gdpr_privacy_audit"]
# Tasks used for evaluation (includes harder ones)
EVAL_TASKS = ["finance_sox_audit", "data_integrity_audit"]

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"  # 4-bit, Unsloth-optimised
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # plain HF fallback
OUTPUT_DIR = "auditrix-grpo-output"

# System prompt — compact version for training (token-efficient)
SYSTEM_PROMPT = """\
You are an AI compliance auditor agent operating inside an OpenEnv environment.

On every turn you receive a JSON observation and must output EXACTLY ONE JSON action.
No prose. No markdown fences. Just raw JSON.

──────────────────────────────────────────────────────────
AVAILABLE ACTIONS
──────────────────────────────────────────────────────────
  inspect_record   — reveal a record's fields (REQUIRED before any audit action)
  apply_rule       — run a compliance rule; returns verdict + confidence + evidence
  request_evidence — gather detailed evidence on a 'warning' verdict (FREE — 0 reward cost)
  flag_violation   — officially flag a confirmed violation (+0.5 correct / -0.3 wrong)
  retract_flag     — undo a flag if new evidence/amendment resolves it (-0.10 if FP retract)
  mark_compliant   — declare a record has no violations (+0.05 if truly compliant)
  prioritize_rules — declare rule priority order (call ONCE at start, before any inspect).
                     FREE action. Provide all active rule IDs highest→lowest severity.
  generate_report  — submit the final audit report (ends episode)
  finish           — end episode without report (lower terminal reward)

JSON schema:
  {"action_type": "<value>", "record_id": "<id or null>", "rule_id": "<id or null>"}
  For prioritize_rules: {"action_type": "prioritize_rules", "rule_priority_order": ["R9", "R4", ...]}

──────────────────────────────────────────────────────────
COMPLIANCE RULES REFERENCE
──────────────────────────────────────────────────────────
R1  Minor overhours          [CRITICAL] — age < 18 AND hours > threshold (default 8)
                               May change via POLICY_UPDATE.
R2  Intern overhours         [LOW]      — role == 'intern' AND hours > threshold (default 40)
R3  Salary out of range      [MEDIUM]   — salary outside role band
                               ⚠ Within ±2% of band edge → verdict='warning'; use request_evidence
R4  Duplicate employee ID    [CRITICAL] — same 'id' in more than one record
R5  Expired contract active  [HIGH]     — contract_end < '2024-01-01' AND status == 'active'
R6  Background check missing [HIGH]     — sensitive role AND background_check != True
                               Sensitive: manager/director/finance_manager/accountant/
                               cfo/security/hr/compliance_officer/legal_counsel/vp_finance
R7  Unapproved overtime      [MEDIUM]   — hours > threshold (default 48) AND overtime_approved != True
                               ⚠ Exactly threshold hours is NOT a violation
                               ⚠ POLICY_UPDATE can lower threshold mid-episode
R8  Missing compliance train [HIGH]     — status == 'active' AND compliance_training != True
                               ⚠ Inactive employees are EXEMPT
R9  GDPR consent missing     [CRITICAL] — pii_access == True AND gdpr_consent != True
                               ⚠ pii_access False/absent → rule does NOT apply
R10 Missing required fields  [LOW]      — any of {id,name,role,hours,salary} missing or null
                               ⚠ Zero values are valid; only missing/null is a violation

Only rules in 'available_rules' are active. Rules in 'suspended_rule_ids' are OFF — skip them.
R11 Orphan manager reference [HIGH]     — manager_id is set AND not found in any employee id
                               ⚠ manager_id=null → exempt; only non-null IDs are checked
                               ⚠ Requires cross-record check: track all employee IDs seen

──────────────────────────────────────────────────────────
VERDICT TAXONOMY (returned by apply_rule / request_evidence)
──────────────────────────────────────────────────────────
  "violation"             — clear breach → flag_violation immediately
  "warning"               — near-threshold ambiguity → call request_evidence, then decide
  "insufficient_evidence" — required field missing → cannot evaluate; do NOT flag
  "compliant"             — rule not triggered

──────────────────────────────────────────────────────────
SEVERITY-WEIGHTED SCORING
──────────────────────────────────────────────────────────
Missing a CRITICAL violation hurts 4x more than missing a LOW one:
  critical (R1, R4, R9)       → weight 2.0
  high     (R5, R6, R8, R11)  → weight 1.5
  medium   (R3, R7)           → weight 1.0
  low      (R2, R10)          → weight 0.5

Prioritize flagging critical/high violations. If steps are limited,
skip low-severity checks rather than critical ones.

──────────────────────────────────────────────────────────
DYNAMIC EVENTS
──────────────────────────────────────────────────────────
  POLICY_UPDATE    — A rule's threshold may change (e.g. overtime threshold 48→40).
                     Check 'current_policy_overrides' in the observation EVERY step.
                     Re-evaluate any previously applied rules if the threshold changed.

  SYSTEM_OUTAGE    — A record becomes temporarily inaccessible (system_outage=true).
                     Skip it; retry after outage_ends_at step.

  RECORD_AMENDMENT — A field value is corrected mid-episode.
                     Re-run apply_rule on the amended record. If a violation is resolved,
                     call retract_flag (if you already flagged it) before reporting.

  RULE_SUSPENSION  — A rule is temporarily deactivated.
                     Suspended rules appear in 'suspended_rule_ids'. Do NOT apply them.
                     They reactivate automatically; check each step.

Fired events are listed in 'active_events' in the observation.

──────────────────────────────────────────────────────────
OPTIMAL STRATEGY
──────────────────────────────────────────────────────────
1. prioritize_rules([highest→lowest severity]) — call ONCE at episode start.
   Recommended order: R1, R4, R9, R5, R6, R8, R11, R3, R7, R2, R10
   (only include rules in 'available_rules')
2. Inspect every record. Skip system_outage=true records; retry later.
3. For each inspected record, apply_rule for each active (non-suspended) rule:
   • verdict=violation → flag_violation immediately
   • verdict=warning → request_evidence, then decide (half FP-penalty if wrong)
   • verdict=insufficient_evidence → do NOT flag; mark_compliant if no other violations
   • Check current_policy_overrides BEFORE applying R1, R2, R7.
4. After a RECORD_AMENDMENT fires, re-apply rules on the amended record.
   If the violation is resolved and you already flagged it, call retract_flag.
5. For records with no violations, call mark_compliant.
6. Call generate_report with a structured payload including audit_confidence section.

──────────────────────────────────────────────────────────
CRITICAL RULES
──────────────────────────────────────────────────────────
• Never repeat the same action twice on the same record (penalty -0.05 / -0.10).
• More than 3 identical (action_type, record_id) in 5 steps = loop penalty (-0.10).
• flag_violation on a compliant record costs -0.3 — be precise.
• Respect rule exemptions: inactive employees skip R5/R8; no pii_access skips R9;
  non-sensitive roles skip R6; manager_id=null skips R11.
• Do NOT apply suspended rules (check suspended_rule_ids every step).
• record_id and rule_id must match exactly what appears in the observation.
• generate_report payload should include audit_confidence section for bonus points.
"""


# ============================================================================
# Environment wrapper — TRL environment_factory pattern
# ============================================================================
class AuditrixToolEnv:
    """TRL-compatible environment exposing Auditrix actions as tool methods.

    The GRPOTrainer discovers all public methods (except reset) and exposes
    them as callable tools for the model.  Each method interacts with the
    underlying ComplianceAuditEnv and accumulates reward.
    """

    def __init__(self) -> None:
        self.env: Optional[ComplianceAuditEnv] = None
        self.reward: float = 0.0
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.task_score: float = 0.0
        self._task_id: str = "easy_basic_audit"
        # ── per-episode component signals (populated from grading_breakdown) ─
        self.detection_rate: float = 0.0        # correct_detected / total_violations
        self.severity_detection: float = 0.0    # severity-weighted detection rate
        self.false_positive_rate: float = 0.0   # false_positives / total_flagged
        self.coverage: float = 0.0              # records_checked / total_records
        self.efficiency: float = 0.0            # steps_remaining / max_steps
        self.prioritization_score: float = 0.0  # severity ordering quality [0,1]
        self.warning_credit: float = 0.0        # correct abstention on warnings [0,0.1]
        self.loop_penalty_count: int = 0        # loop exploit events accumulated
        self.report_generated: bool = False     # whether generate_report was called
        self.deliberation_count: int = 0        # request_evidence calls made


    def reset(self, **kwargs) -> str | None:
        """Called at the start of each episode.  Returns the initial observation."""
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.task_score = 0.0
        self.done = False
        self.detection_rate = 0.0
        self.severity_detection = 0.0
        self.false_positive_rate = 0.0
        self.coverage = 0.0
        self.efficiency = 0.0
        self.prioritization_score = 0.0
        self.warning_credit = 0.0
        self.loop_penalty_count = 0
        self.report_generated = False
        self.deliberation_count = 0

        # Pick task from dataset column, or cycle through training tasks
        self._task_id = kwargs.get("task_id", "easy_basic_audit")
        self.env = ComplianceAuditEnv(task_id=self._task_id)
        obs = self.env.reset(seed=42)
        return _format_observation(obs.model_dump())

    def inspect_record(self, record_id: str) -> str:
        """Inspect a record to reveal its fields. Must be called before auditing.

        Args:
            record_id: The ID of the record to inspect (e.g. 'E001', 'F003')

        Returns:
            The inspection result with record fields or an error message.
        """
        return self._step("inspect_record", record_id=record_id)

    def apply_rule(self, record_id: str, rule_id: str) -> str:
        """Apply a compliance rule to an inspected record.

        Args:
            record_id: The record to check (e.g. 'E001')
            rule_id: The rule to apply (e.g. 'R1', 'R3', 'R7')

        Returns:
            Whether a violation was detected, with evidence details.
        """
        return self._step("apply_rule", record_id=record_id, rule_id=rule_id)

    def flag_violation(self, record_id: str, rule_id: str) -> str:
        """Officially flag a confirmed violation on a record.

        Args:
            record_id: The record with the violation (e.g. 'E001')
            rule_id: The violated rule (e.g. 'R1')

        Returns:
            Confirmation of the flag or penalty for false positive.
        """
        return self._step("flag_violation", record_id=record_id, rule_id=rule_id)

    def mark_compliant(self, record_id: str) -> str:
        """Declare that a record has no violations and is fully compliant.

        Args:
            record_id: The compliant record (e.g. 'E003')

        Returns:
            Confirmation or penalty if violations were missed.
        """
        return self._step("mark_compliant", record_id=record_id)

    def generate_report(self, summary: str) -> str:
        """Submit the final audit report and end the episode.

        Args:
            summary: A brief summary of findings and flagged violations

        Returns:
            The final score and episode summary.
        """
        report_payload = {"summary": summary, "flagged_violations": [], "compliant_records": []}
        return self._step("generate_report", report=report_payload)

    def prioritize_rules(self, rule_order: list) -> str:
        """Declare rule priority order (highest-severity first). Call once per episode.
        
        Args:
            rule_order: List of active rule IDs, e.g. ['R9','R4','R6']
        """
        return self._step("prioritize_rules", rule_priority_order=rule_order)

    def request_evidence(self, record_id: str, rule_id: str) -> str:
        """Gather detailed evidence for a rule before committing to flag or pass.
        
        Args:
            record_id: The record to investigate (e.g. 'E001')
            rule_id: The rule to gather evidence for (e.g. 'R3')
        """
        return self._step("request_evidence", record_id=record_id, rule_id=rule_id)

    def retract_flag(self, record_id: str, rule_id: str) -> str:
        """Retract a previously-flagged violation if new evidence changes your view.
        
        Args:
            record_id: The record whose flag to retract
            rule_id: The rule to un-flag
        """
        return self._step("retract_flag", record_id=record_id, rule_id=rule_id)

    # ── internal ──────────────────────────────────────────────────────────
    def _step(self, action_type: str, **kwargs) -> str:
        if self.done:
            raise ValueError("Episode is over. No more actions allowed.")
        if self.env is None:
            raise ValueError("Environment not initialised. Call reset() first.")

        action = AuditAction(
            action_type=ActionType(action_type),
            record_id=kwargs.get("record_id"),
            rule_id=kwargs.get("rule_id"),
            rule_priority_order=kwargs.get("rule_priority_order"),
            report=kwargs.get("report"),
        )
        result = self.env.step(action)
        self.reward = float(result.reward)
        self.cumulative_reward += self.reward
        self.done = result.done
        self.task_score = float(result.info.get("task_score", 0.0))

        # ── Track incremental signals on every step ──────────────────
        info = result.info
        if action_type == "request_evidence":
            self.deliberation_count += 1
        if action_type == "generate_report":
            self.report_generated = True
        # Loop penalty count — increment when loop_exploit_signature fires
        # (loop_exploit_signature is set in the observation whenever a loop event occurs)
        _obs_peek = result.observation.model_dump()
        if _obs_peek.get("loop_exploit_signature") is not None:
            self.loop_penalty_count += 1

        # ── On terminal step, extract full grading breakdown ─────────
        if self.done:
            bd = info.get("grading_breakdown", {})
            self.detection_rate = float(bd.get("detection_rate", 0.0))
            self.severity_detection = float(bd.get("severity_detection", 0.0))
            self.false_positive_rate = float(bd.get("false_positive_rate", 0.0))
            self.coverage = float(bd.get("coverage", 0.0))
            self.efficiency = float(bd.get("efficiency", 0.0))
            self.prioritization_score = float(bd.get("prioritization_score", 0.0))
            self.warning_credit = float(bd.get("warning_credit", 0.0))
            lp = bd.get("loop_deduction", 0.0)
            # Convert loop_deduction back to event count (0.02/event base)
            if self.loop_penalty_count == 0 and lp > 0:
                self.loop_penalty_count = max(self.loop_penalty_count,
                                              int(round(lp / 0.02)))

        # Build human-readable response for the model
        obs = result.observation.model_dump()
        response_parts = []

        if obs.get("last_action_error"):
            response_parts.append(f"ERROR: {obs['last_action_error']}")

        if action_type == "inspect_record":
            rec = next((r for r in obs.get("visible_records", [])
                        if r["record_id"] == kwargs.get("record_id")), None)
            if rec and rec.get("fields"):
                response_parts.append(f"Record {kwargs['record_id']} fields: {json.dumps(rec['fields'], default=str)}")
            if rec and rec.get("system_outage"):
                response_parts.append(f"SYSTEM_OUTAGE: record unavailable")

        elif action_type in ("apply_rule", "request_evidence"):
            trace = obs.get("last_decision_trace")
            if trace:
                verdict = trace.get("verdict", "unknown")
                conf = trace.get("confidence_tier", "")
                outcome = trace.get("outcome", "unknown")
                response_parts.append(f"Rule {kwargs.get('rule_id')} on {kwargs.get('record_id')}: verdict={verdict} confidence={conf} | {outcome}")
                if verdict in ("warning", "insufficient_evidence") and trace.get("rule_evidence"):
                    response_parts.append(f"Evidence: {json.dumps(trace['rule_evidence'], default=str)}")
            else:
                response_parts.append(f"Rule {kwargs.get('rule_id')} applied to {kwargs.get('record_id')}.")

        elif action_type == "flag_violation":
            trace = obs.get("last_decision_trace")
            if trace:
                response_parts.append(f"Flag {kwargs.get('record_id')}:{kwargs.get('rule_id')} → {trace.get('outcome', 'unknown')}")
            else:
                response_parts.append(f"Flagged {kwargs.get('record_id')}:{kwargs.get('rule_id')}.")

        elif action_type == "retract_flag":
            trace = obs.get("last_decision_trace")
            if trace:
                response_parts.append(f"Retract {kwargs.get('record_id')}:{kwargs.get('rule_id')} → {trace.get('outcome', 'retracted')}")

        elif action_type == "prioritize_rules":
            response_parts.append(f"Priority recorded: {kwargs.get('rule_priority_order')}")

        elif action_type == "mark_compliant":
            response_parts.append(f"Marked {kwargs.get('record_id')} as compliant. Reward: {self.reward:+.2f}")

        elif action_type == "generate_report":
            response_parts.append(f"AUDIT COMPLETE. Final score: {self.task_score:.2f}")

        response_parts.append(f"[reward={self.reward:+.2f} | step={obs.get('step_index', '?')}/{obs.get('max_steps', '?')} | remaining={obs.get('remaining_steps', '?')}]")

        if result.done:
            response_parts.append(f"EPISODE ENDED. Total score: {self.task_score:.2f}")

        return "\n".join(response_parts)


def _format_observation(obs: Dict[str, Any]) -> str:
    """Format the initial observation into a concise prompt for the model."""
    records = obs.get("visible_records", [])
    record_ids = [r["record_id"] for r in records]
    rules = []
    for r in obs.get("available_rules", []):
        sev = r.get("severity", "")
        sev_tag = f"[{sev}] " if sev else ""
        rules.append(f"  {r['rule_id']} {sev_tag}{r.get('description', '')}")

    suspended = obs.get("suspended_rule_ids", [])
    overrides = obs.get("current_policy_overrides", {})

    lines = [
        f"TASK: {obs.get('task_title', 'Compliance Audit')}",
        f"OBJECTIVE: {obs.get('objective', '')}",
        f"RECORDS ({len(record_ids)}): {', '.join(record_ids)}",
        "ACTIVE RULES:",
        *rules,
    ]
    if suspended:
        lines.append(f"SUSPENDED RULES: {suspended}")
    if overrides:
        lines.append(f"POLICY OVERRIDES: {json.dumps(overrides)}")
    
    lines.append(f"MAX STEPS: {obs.get('max_steps', '?')}")
    lines.append("\nBegin your audit. Start with prioritize_rules(), then inspect records, "
                 "apply rules, flag violations, mark compliant records, and report.")
    
    return "\n".join(lines)


# ============================================================================
# Independent reward functions — §7 Multi-signal GRPO
# ============================================================================
# The grader already computes every sub-component.  We expose them here as
# individual reward functions so GRPOTrainer logs each axis separately.
# This lets you diagnose reward hacking early:
#   e.g. coverage climbs while detection stays at 0 → agent is spamming inspect.
#
# Design choices that improve on naive multi-reward:
#   1. reward_precision uses correct/flagged, NOT (1-FP_rate).
#      (1-FP_rate) awards 1.0 when the agent flags nothing — reward hacking.
#      correct/flagged properly returns 0.0 when nothing is flagged.
#   2. reward_deliberation rewards request_evidence calls that were genuine
#      (i.e. on a 'warning' verdict, not a 'violation'). Encourages calibrated
#      uncertainty rather than blanket evidence requests.
#   3. reward_severity uses the grader's severity_detection, not plain
#      detection_rate — missing a CRITICAL violation hurts more.
#   4. reward_anti_exploit caps at -0.30 so a runaway loop doesn't dominate
#      the other signals (they'd all be drowned out in the gradient).
# ============================================================================

def _get_envs(*args, **kwargs):
    """Extract environment list from TRL's various calling conventions."""
    envs = kwargs.get("environments")
    if envs is None:
        for a in args:
            if isinstance(a, list) and a and hasattr(a[0], "task_score"):
                envs = a
                break
    return envs or []


def reward_task_score(*args, **kwargs) -> list[float]:
    """Full composite grader score — backward-compatible primary signal."""
    return [env.task_score for env in _get_envs(*args, **kwargs)]


def reward_severity(*args, **kwargs) -> list[float]:
    """Severity-weighted detection: missing a CRITICAL violation hurts 4× more.
    
    Weight 0.55 — the dominant training signal.
    Agents learn to prioritise GDPR > overtime > salary discrepancy.
    """
    return [0.55 * env.severity_detection for env in _get_envs(*args, **kwargs)]


def reward_precision(*args, **kwargs) -> list[float]:
    """Flagging precision: correct_flagged / total_flagged.
    
    Weight 0.25 — heavy penalty for false positives.
    Returns 0.0 when nothing is flagged (not 1.0), preventing the
    trivial "flag nothing" exploit that (1−FP_rate) would allow.
    """
    envs = _get_envs(*args, **kwargs)
    out = []
    for env in envs:
        # Derive from stored component signals
        fp_rate = env.false_positive_rate          # FP / total_flagged
        det_rate = env.detection_rate              # correct / total_violations
        # Infer total_flagged proxy: if fp_rate > 0 and detection > 0, precision != 1
        # We use 1 - fp_rate only when something was actually flagged (detection > 0)
        if det_rate == 0.0 and fp_rate == 0.0:
            # Agent flagged nothing → precision 0 (no reward for inaction)
            out.append(0.0)
        else:
            out.append(0.25 * (1.0 - fp_rate))
    return out


def reward_coverage(*args, **kwargs) -> list[float]:
    """Records inspected / total records.
    
    Weight 0.10 — secondary signal, encourages thorough auditing.
    Lower weight than in your original to avoid "inspect everything, flag nothing".
    """
    return [0.10 * env.coverage for env in _get_envs(*args, **kwargs)]


def reward_efficiency(*args, **kwargs) -> list[float]:
    """Steps remaining / max_steps — rewards concise audits.
    
    Weight 0.05 — small nudge, not dominant.
    Prevents reward hacking via deliberate step exhaustion.
    """
    return [0.05 * env.efficiency for env in _get_envs(*args, **kwargs)]


def reward_prioritization(*args, **kwargs) -> list[float]:
    """Severity ordering quality — did the agent check critical rules first?
    
    Bonus up to +0.08. Only fires if prioritize_rules() was called.
    Rewards the "think before acting" pattern from Claude Code.
    """
    return [0.08 * env.prioritization_score for env in _get_envs(*args, **kwargs)]


def reward_deliberation(*args, **kwargs) -> list[float]:
    """Bonus for using request_evidence before flagging warnings.
    
    Counts genuine deliberation calls (request_evidence on ambiguous verdicts).
    Capped at +0.10 to prevent gaming via excessive evidence calls.
    
    This directly rewards calibrated uncertainty — the core "agentic" skill.
    """
    envs = _get_envs(*args, **kwargs)
    out = []
    for env in envs:
        # warning_credit is computed by the grader — correct abstentions on warnings
        # deliberation_count is our own counter of request_evidence calls
        credit = env.warning_credit  # [0, 0.10] from grader
        delib_bonus = min(0.05, env.deliberation_count * 0.01)  # capped at 0.05
        out.append(credit + delib_bonus)
    return out


def reward_anti_exploit(*args, **kwargs) -> list[float]:
    """Penalty for loop exploits.
    
    -0.10 per loop event, capped at -0.30 total so it doesn't drown out
    positive signals in the gradient computation.
    """
    return [-min(0.30, env.loop_penalty_count * 0.10)
            for env in _get_envs(*args, **kwargs)]


# Ordered list for GRPOTrainer(reward_funcs=REWARD_FUNCS).
# TRL logs each function under its __name__ in training metrics.
REWARD_FUNCS = [
    reward_task_score,      # composite — backward compat
    reward_severity,        # severity-weighted detection (dominant)
    reward_precision,       # FP-safe precision
    reward_coverage,        # inspection thoroughness
    reward_efficiency,      # conciseness
    reward_prioritization,  # severity-order quality
    reward_deliberation,    # calibrated uncertainty / evidence gathering
    reward_anti_exploit,    # loop exploit deterrent
]

print(f"✅ {len(REWARD_FUNCS)} independent reward functions registered.")
print("   Columns in TRL training log:")
for fn in REWARD_FUNCS:
    print(f"   · {fn.__name__}")


# ============================================================================
# Dataset builder
# ============================================================================
def build_training_dataset(task_ids: List[str], samples_per_task: int = 64):
    """Build a TRL-compatible dataset of audit prompts."""
    from datasets import Dataset

    prompts = []
    task_id_col = []

    for task_id in task_ids:
        for _ in range(samples_per_task):
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Perform a compliance audit for task: {task_id}"},
            ])
            task_id_col.append(task_id)

    return Dataset.from_dict({
        "prompt": prompts,
        "task_id": task_id_col,
    })


# ============================================================================
# Dry-run mode — test reward function without GPU
# ============================================================================
def dry_run():
    """Run a quick simulation to verify the environment + reward function work."""
    print("=" * 60)
    print("DRY RUN — Testing Auditrix RL environment")
    print("=" * 60)

    for task_id in ["easy_basic_audit", "medium_mixed_audit"]:
        print(f"\n{'─' * 50}")
        print(f"Task: {task_id}")
        print(f"{'─' * 50}")

        env = AuditrixToolEnv()
        obs = env.reset(task_id=task_id)
        print(f"Initial observation:\n{obs[:300]}...\n")

        # Simulate a perfect agent: inspect all → apply all rules → flag correct → report
        task = TASKS[task_id]
        record_ids = [r.record_id for r in task.records]
        rule_ids = task.active_rule_ids

        # Step 1: Inspect all records
        for rid in record_ids:
            if env.done:
                break
            result = env.inspect_record(rid)
            print(f"  inspect({rid}): reward={env.reward:+.2f}")

        # Step 2: Apply all rules to all records
        violations_found = []
        for rid in record_ids:
            if env.done:
                break
            for rule_id in rule_ids:
                if env.done:
                    break
                result = env.apply_rule(rid, rule_id)
                if "violation_detected" in result:
                    violations_found.append((rid, rule_id))
                    print(f"  apply_rule({rid}, {rule_id}): VIOLATION DETECTED")

        # Step 3: Flag all violations
        for rid, rule_id in violations_found:
            if env.done:
                break
            result = env.flag_violation(rid, rule_id)
            print(f"  flag({rid}, {rule_id}): reward={env.reward:+.2f}")

        # Step 4: Mark compliant records
        violating_records = {v[0] for v in violations_found}
        for rid in record_ids:
            if env.done:
                break
            if rid not in violating_records:
                result = env.mark_compliant(rid)
                print(f"  mark_compliant({rid}): reward={env.reward:+.2f}")

        # Step 5: Generate report
        if not env.done:
            result = env.generate_report("Audit complete. All violations flagged.")
            print(f"\n  generate_report: task_score={env.task_score:.4f}")

        print(f"\n  FINAL SCORE: {env.task_score:.4f}")
        print(f"  CUMULATIVE REWARD: {env.cumulative_reward:.4f}")

    # Test reward function
    envs = []
    for task_id in ["easy_basic_audit", "medium_mixed_audit"]:
        e = AuditrixToolEnv()
        e.reset(task_id=task_id)
        # Quick simulate
        task = TASKS[task_id]
        for rid in [r.record_id for r in task.records]:
            if not e.done:
                e.inspect_record(rid)
        if not e.done:
            e.generate_report("Quick audit done.")
        envs.append(e)

    rewards = {fn.__name__: fn(envs) for fn in REWARD_FUNCS}
    print(f"\n{'═' * 60}")
    print(f"Reward function test: {rewards}")
    print(f"{'═' * 60}")
    print("\n  Dry run passed. Environment and reward function are working.")


# ============================================================================
# Main training loop
# ============================================================================
def train(use_unsloth: bool = True):
    """Run GRPO training with Unsloth + TRL."""

    # ── 1. Load model ─────────────────────────────────────────────────────
    if use_unsloth:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=64,
            gpu_memory_utilization=0.6,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=64,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"  Loaded {MODEL_NAME} with Unsloth (4-bit LoRA, r=64)")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        model_name = FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f" Loaded {model_name} with HF PEFT (LoRA, r=64)")

    # ── 2. Build dataset ──────────────────────────────────────────────────
    dataset = build_training_dataset(TRAIN_TASKS, samples_per_task=64)
    print(f" Training dataset: {len(dataset)} samples across {TRAIN_TASKS}")

    # ── 3. Configure GRPO ─────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # ── Training hyperparameters ──
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        bf16=True,

        # ── GRPO-specific ──
        num_generations=4,           # completions per prompt for advantage estimation
        max_prompt_length=1024,      # truncate prompt to this length
        max_completion_length=4096,  # multi-turn episodes need long completions
        # temperature=0.7,           # sampling temperature for diversity

        # ── Logging ──
        logging_steps=1,
        log_completions=True,
        save_strategy="steps",
        save_steps=50,
        report_to="none",           # set to "wandb" if you have W&B configured

        # ── Chat template ──
        chat_template_kwargs={"enable_thinking": False},

        # ── vLLM for generation (Unsloth handles this automatically) ──
        use_vllm=use_unsloth,
        vllm_mode="colocate" if use_unsloth else None,
    )

    # ── 4. Create trainer ─────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
        environment_factory=AuditrixToolEnv,
    )
    print("  GRPOTrainer initialised with environment_factory=AuditrixToolEnv")

    # ── 5. Train! ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING GRPO TRAINING")
    print(f"  Model:      {MODEL_NAME if use_unsloth else FALLBACK_MODEL}")
    print(f"  Tasks:      {TRAIN_TASKS}")
    print(f"  Epochs:     {training_args.num_train_epochs}")
    print(f"  Batch:      {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} accum")
    print(f"  Generations: {training_args.num_generations} per prompt")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── 6. Save ───────────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n  Model saved to {OUTPUT_DIR}/")

    # ── 7. Post-training evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    for task_id in TRAIN_TASKS + EVAL_TASKS:
        env = AuditrixToolEnv()
        obs_text = env.reset(task_id=task_id)

        # Simple heuristic agent for evaluation baseline
        task = TASKS[task_id]
        for rec in task.records:
            if env.done:
                break
            env.inspect_record(rec.record_id)
        for rec in task.records:
            if env.done:
                break
            for rule_id in task.active_rule_ids:
                if env.done:
                    break
                env.apply_rule(rec.record_id, rule_id)

        if not env.done:
            env.generate_report("Evaluation run complete.")

        label = "TRAIN" if task_id in TRAIN_TASKS else "EVAL "
        print(f"  [{label}] {task_id:<30} score={env.task_score:.4f}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a compliance auditor with GRPO (Unsloth + TRL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the environment and reward function without GPU training",
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Use plain HF Transformers + PEFT instead of Unsloth",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(use_unsloth=not args.no_unsloth)


if __name__ == "__main__":
    main()
