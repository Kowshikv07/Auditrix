"""
Auditrix GRPO Training — Easy difficulty only
Single-turn architecture: model sees all records upfront, outputs one JSON audit report.
"""
from __future__ import annotations
import json, os, re, torch, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_TASKS      = ["easy_basic_audit"]
SAMPLES_PER_TASK = 64
GRPO_STEPS       = 200
LORA_RANK        = 16
OUTPUT_DIR       = "auditrix-grpo-output"
HF_TOKEN         = os.getenv("HF_TOKEN")
PUSH_TO_HUB      = os.getenv("PUSH_TO_HUB", "0") == "1"
HUB_REPO_ID      = os.getenv("HUB_REPO_ID", "")

# ── Health-check server ───────────────────────────────────────────────────────
_TRAIN_STATUS = {"step": 0, "total": GRPO_STEPS, "done": False}

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        s = _TRAIN_STATUS
        body = (
            f"<h2>Auditrix GRPO Training</h2>"
            f"<p>Step: {s['step']}/{s['total']}</p>"
            f"<p>{'✅ Done' if s['done'] else '🔄 Training...'}</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass

def _start_health_server(port: int = 7860):
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Health server on port {port}")

# ── Imports ───────────────────────────────────────────────────────────────────
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from openenv_compliance_audit.environment import ComplianceAuditEnv
from openenv_compliance_audit.models import AuditAction, ActionType
from openenv_compliance_audit.tasks import TASKS

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Auditrix, an expert compliance auditor that reasons under uncertainty.

TOOL WORKFLOW:
1. inspect_record(record_id)            — reveal a record's fields (required first)
2. prioritize_rules(rule_order)         — declare your intended rule priority (highest-severity first)
3. apply_rule(record_id, rule_id)       — test a rule; returns verdict + confidence + evidence
4. request_evidence(record_id, rule_id) — gather evidence before committing (free action)
5. flag_violation(record_id, rule_id)   — officially flag a confirmed violation
6. retract_flag(record_id, rule_id)     — undo a previous flag if new evidence changes your view
7. mark_compliant(record_id)            — declare a record has no violations
8. generate_report(summary)             — submit final audit report (ends episode)

VERDICT TAXONOMY returned by apply_rule / request_evidence:
  "violation"             — clear breach; flag immediately
  "warning"               — near-threshold; gather more evidence before flagging
  "insufficient_evidence" — required fields missing; cannot evaluate safely
  "compliant"             — rule not triggered

RULES (only active rules apply per task):
R1  [critical]: age<18 and hours>8 → minor employee overhours
R2  [low]:      role='intern' and hours>40 → intern overhours
R3  [medium]:   salary outside role band → salary violation (±2% = warning zone)
R4  [critical]: duplicate employee ID across records
R5  [high]:     contract_end<'2024-01-01' and status='active' → expired contract
R6  [high]:     sensitive role without background_check → missing background check
R7  [medium]:   hours>48 and overtime_approved!=True → unapproved overtime
R8  [high]:     status='active' and compliance_training!=True → missing training
R9  [critical]: pii_access=True and gdpr_consent!=True → GDPR consent missing
R10 [low]:      missing required fields (id, name, role, hours, salary)
R11 [high]:     manager_id references non-existent employee (cross-record check)

SEVERITY-BASED SCORING: critical/high violations are worth MORE. Prioritize them.
DYNAMIC EVENTS may change rules mid-episode — always use apply_rule with current state.

STRATEGY:
  1. prioritize_rules([highest→lowest severity]) once at the start
  2. For each record: inspect → apply high-severity rules first
     - verdict=violation → flag_violation immediately
     - verdict=warning → request_evidence, then decide flag vs pass
     - verdict=insufficient_evidence → mark_compliant if no other violations
  3. If a POLICY_UPDATE fires, re-apply affected rules on already-inspected records
  4. If you flagged in error, retract_flag before report
  5. generate_report when done or steps are running low

All employee record fields are already revealed below.
Analyze every record against every active rule, then output ONE JSON object only.
No prose. No markdown fences. Raw JSON only.

Output format:
{
  "violations": [
    {"record_id": "E001", "rule_id": "R1", "reason": "brief explanation"}
  ],
  "compliant_records": ["E002", "E003"],
  "summary": "Found N violations across M records."
}

Each violation MUST use exactly these keys: "record_id", "rule_id", "reason".
Be precise — false positives are penalised heavily."""

# ── Ground truth ──────────────────────────────────────────────────────────────
def get_ground_truth(task_id: str) -> set:
    env = ComplianceAuditEnv(task_id=task_id)
    env.reset(seed=42)
    task = TASKS[task_id]
    violations = set()
    for rec in task.records:
        env.step(AuditAction(action_type=ActionType("inspect_record"),
                             record_id=rec.record_id))
    for rec in task.records:
        for rule_id in task.active_rule_ids:
            result = env.step(AuditAction(action_type=ActionType("apply_rule"),
                                          record_id=rec.record_id, rule_id=rule_id))
            trace = result.observation.model_dump().get("last_decision_trace") or {}
            if trace.get("outcome") in ("violation", "violation_detected"):
                violations.add((rec.record_id, rule_id))
    env.close()
    return violations

print("Computing ground truth...")
GROUND_TRUTH: dict[str, set] = {t: get_ground_truth(t) for t in TRAIN_TASKS}
for t, v in GROUND_TRUTH.items():
    print(f"  {t}: {len(v)} violations → {sorted(v)}")

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(task_id: str) -> list[dict]:
    task = TASKS[task_id]
    env  = ComplianceAuditEnv(task_id=task_id)
    obs  = env.reset(seed=42)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else {}

    revealed = {}
    for rec in task.records:
        result = env.step(AuditAction(action_type=ActionType("inspect_record"),
                                      record_id=rec.record_id))
        obs_d = result.observation.model_dump() if hasattr(result.observation, "model_dump") else {}
        for r in obs_d.get("visible_records", []):
            if r.get("record_id") == rec.record_id:
                revealed[rec.record_id] = r.get("fields", {})
    env.close()

    available_rules = obs_dict.get("available_rules") or []
    if available_rules:
        rules_text = "\n".join(
            f"  {r['rule_id']}: {r.get('description', '')}" for r in available_rules)
    else:
        rules_text = "\n".join(
            f"  {rid}: (see RULES REFERENCE above)" for rid in task.active_rule_ids)

    user_content = (
        f"TASK: {task_id}\n\n"
        f"ACTIVE RULES:\n{rules_text}\n\n"
        f"EMPLOYEE RECORDS:\n{json.dumps(revealed, indent=2, default=str)}\n\n"
        f"Output your JSON audit report now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

# ── Reward helpers ────────────────────────────────────────────────────────────
def _parse_output(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {}

_RID_KEYS  = ("record_id", "employee_id", "id", "emp_id", "employee", "record")
_RULE_KEYS = ("rule_id", "rule", "rule_name", "violation_rule",
              "violated_rule", "violated_rule_id", "ruleid")

def _extract_violations(output: dict) -> set[tuple[str, str]]:
    predicted = set()
    for v in output.get("violations", []):
        if not isinstance(v, dict): continue
        rid  = next((v[k] for k in _RID_KEYS  if v.get(k)), None)
        rule = next((v[k] for k in _RULE_KEYS if v.get(k)), None)
        if rid and rule:
            predicted.add((str(rid), str(rule)))
    return predicted

def _score(output: dict, task_id: str) -> tuple[float, float, float]:
    gt        = GROUND_TRUTH.get(task_id, set())
    predicted = _extract_violations(output)
    if not gt and not predicted: return 1.0, 1.0, 1.0
    if not gt:                   return 1.0, 0.0, 0.0
    if not predicted:            return 0.0, 0.0, 0.0
    tp        = len(predicted & gt)
    fp        = len(predicted - gt)
    detection = tp / len(gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1        = (2 * precision * detection / (precision + detection)
                 if (precision + detection) > 0 else 0.0)
    return detection, precision, f1

# ── Shared log store (written by reward_f1, read by callback) ─────────────────
_LAST_BATCH: list[dict] = []

def reward_f1(completions, prompts, task_id, **kwargs) -> list[float]:
    global _LAST_BATCH
    _LAST_BATCH = []
    scores = []
    for completion, prompt, tid in zip(completions, prompts, task_id):
        text   = completion[0]["content"] if isinstance(completion, list) else str(completion)
        parsed = _parse_output(text)
        det, prec, f1 = _score(parsed, tid)

        # Coverage = fraction of GT violations found (same as detection)
        coverage = det

        _LAST_BATCH.append({
            "prompt"    : prompt[-1]["content"][:120].replace("\n", " ") + "…",
            "completion": text[:200].replace("\n", " ") + "…",
            "f1"        : round(f1,       3),
            "coverage"  : round(coverage, 3),
        })
        scores.append(float(f1))
    return scores

def reward_format(completions, prompts, **kwargs) -> list[float]:
    scores = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        out  = _parse_output(text)
        ok   = all(k in out for k in ("violations", "compliant_records", "summary"))
        scores.append(0.1 if ok else 0.0)
    return scores

def reward_detection(completions, prompts, task_id, **kwargs) -> list[float]:
    scores = []
    for completion, tid in zip(completions, task_id):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        det, _, _ = _score(_parse_output(text), tid)
        scores.append(float(det) * 0.6)
    return scores

def reward_precision(completions, prompts, task_id, **kwargs) -> list[float]:
    scores = []
    for completion, tid in zip(completions, task_id):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        _, prec, _ = _score(_parse_output(text), tid)
        scores.append(float(prec) * 0.4)
    return scores

# ── Clean logging callback ────────────────────────────────────────────────────
class AuditLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step    = state.global_step
        reward  = logs.get("reward")
        _TRAIN_STATUS["step"] = step

        if reward is None: return  # skip non-reward log lines

        print(f"\n{'═'*72}")
        print(f"Step {step:>4}/{args.max_steps}  |  reward_from_env={reward:.4f}")
        print(f"{'─'*72}")

        for i, entry in enumerate(_LAST_BATCH):
            # Advantage = reward - mean reward across batch (approximation for display)
            batch_mean = sum(e["f1"] for e in _LAST_BATCH) / len(_LAST_BATCH)
            advantage  = entry["f1"] - batch_mean

            print(f"  [{i+1}] Prompt    : {entry['prompt']}")
            print(f"       Completion: {entry['completion']}")
            print(f"       reward_from_env={entry['f1']:.4f} | "
                  f"reward_format={0.1 if entry['f1'] > 0 else 0.0:.1f} | "
                  f"reward_coverage={entry['coverage']:.4f} | "
                  f"Advantage={advantage:+.4f}")
            print()

    def on_train_end(self, args, state, control, **kwargs):
        _TRAIN_STATUS["done"] = True
        print("\n✅ Training complete.")

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"CUDA: {torch.cuda.is_available()}" +
          (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb,
        device_map="auto", token=HF_TOKEN, trust_remote_code=True)
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    model.print_trainable_parameters()
    return model, tokenizer

# ── Dataset ───────────────────────────────────────────────────────────────────
def build_dataset() -> Dataset:
    prompts, task_ids = [], []
    for task_id in TRAIN_TASKS:
        prompt = build_prompt(task_id)
        for _ in range(SAMPLES_PER_TASK):
            prompts.append(prompt)
            task_ids.append(task_id)
    ds = Dataset.from_dict({"prompt": prompts, "task_id": task_ids})
    print(f"Dataset: {len(ds)} samples | tasks: {TRAIN_TASKS}")
    return ds

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    model, tokenizer = load_model()
    dataset = build_dataset()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    config = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1800,
        max_completion_length=768,
        max_steps=GRPO_STEPS,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_f1, reward_format, reward_detection, reward_precision],
        args=config,
        train_dataset=dataset,
        callbacks=[AuditLogCallback()],
    )

    print(f"\nStarting GRPO — {GRPO_STEPS} steps | task: easy_basic_audit only")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved → {OUTPUT_DIR}/")

    if PUSH_TO_HUB and HUB_REPO_ID:
        model.push_to_hub(HUB_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HUB_REPO_ID, token=HF_TOKEN)
        print(f"Pushed → {HUB_REPO_ID}")

    print("\nGround truth used during training:")
    for task_id in TRAIN_TASKS:
        gt = GROUND_TRUTH[task_id]
        print(f"  {task_id}: {len(gt)} violations → {sorted(gt)}")

if __name__ == "__main__":
    _start_health_server(port=7860)
    train()