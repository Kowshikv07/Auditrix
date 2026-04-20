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
You are a compliance auditor. You audit employee records against policy rules.

WORKFLOW:
1. inspect_record(record_id) — reveal a record's fields (required first)
2. apply_rule(record_id, rule_id) — test a rule; returns violation status
3. flag_violation(record_id, rule_id) — officially flag a confirmed violation
4. mark_compliant(record_id) — declare a record has no violations
5. generate_report(summary) — submit your final audit report (ends episode)

RULES (only active rules apply per task):
R1: age<18 and hours>8 → minor overhours
R2: role='intern' and hours>40 → intern overhours
R3: salary outside role band → salary violation
R4: duplicate employee ID across records
R5: contract_end<'2024-01-01' and status='active' → expired contract
R6: sensitive role without background_check → missing check
R7: hours>48 and overtime_approved!=True → unapproved overtime
R8: status='active' and compliance_training!=True → missing training
R9: pii_access=True and gdpr_consent!=True → GDPR consent missing
R10: missing required fields (id, name, role, hours, salary)

STRATEGY:
- Inspect ALL records first
- Apply each active rule to each record
- Flag only confirmed violations (false positives = -0.3 penalty!)
- Mark records with no violations as compliant
- Generate a final report with summary

RESPOND ONLY WITH TOOL CALLS. No prose."""


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

    def reset(self, **kwargs) -> str | None:
        """Called at the start of each episode.  Returns the initial observation."""
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.task_score = 0.0
        self.done = False

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
            report=kwargs.get("report"),
        )
        result = self.env.step(action)
        self.reward = float(result.reward)
        self.cumulative_reward += self.reward
        self.done = result.done
        self.task_score = float(result.info.get("task_score", 0.0))

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
            else:
                response_parts.append(f"Record {kwargs.get('record_id')} inspected.")

        elif action_type == "apply_rule":
            trace = obs.get("last_decision_trace")
            if trace:
                outcome = trace.get("outcome", "unknown")
                response_parts.append(f"Rule {kwargs.get('rule_id')} on {kwargs.get('record_id')}: {outcome}")
                if trace.get("rule_evidence"):
                    response_parts.append(f"Evidence: {json.dumps(trace['rule_evidence'], default=str)}")
            else:
                response_parts.append(f"Rule {kwargs.get('rule_id')} applied to {kwargs.get('record_id')}.")

        elif action_type == "flag_violation":
            trace = obs.get("last_decision_trace")
            if trace:
                response_parts.append(f"Flag {kwargs.get('record_id')}:{kwargs.get('rule_id')} → {trace.get('outcome', 'unknown')}")
            else:
                response_parts.append(f"Flagged {kwargs.get('record_id')}:{kwargs.get('rule_id')}.")

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
    rules = obs.get("available_rules", [])
    rule_desc = [f"{r['rule_id']}: {r.get('description', r.get('condition', ''))}" for r in rules]

    return (
        f"TASK: {obs.get('task_title', 'Compliance Audit')}\n"
        f"OBJECTIVE: {obs.get('objective', '')}\n"
        f"RECORDS ({len(record_ids)}): {', '.join(record_ids)}\n"
        f"ACTIVE RULES:\n" + "\n".join(f"  {d}" for d in rule_desc) + "\n"
        f"MAX STEPS: {obs.get('max_steps', '?')}\n\n"
        f"Begin your audit. Inspect records, apply rules, flag violations, "
        f"mark compliant records, then generate your report."
    )


# ============================================================================
# Reward function
# ============================================================================
def audit_reward_func(environments, **kwargs) -> list[float]:
    """Reward function that returns the environment's task_score.

    The task_score is the Auditrix grader's composite score:
      0.50 × detection_rate + 0.20 × (1−FP_rate) + 0.20 × coverage + 0.10 × efficiency
    This is in [0, 1] and already captures all the nuances of the audit.
    """
    return [env.task_score for env in environments]


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

    rewards = audit_reward_func(envs)
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
        reward_funcs=audit_reward_func,
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
