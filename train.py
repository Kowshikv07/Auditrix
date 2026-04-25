"""
Auditrix GRPO Training — Easy difficulty only
Single-turn architecture: model sees all records upfront, outputs one JSON audit report.
"""
from __future__ import annotations
import json, os, re, torch, threading
from html import escape
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
_TRAIN_STATUS = {
    "step": 0,
    "total": GRPO_STEPS,
    "done": False,
    "last_reward": 0.0,
    "rows": [],
}


def _render_rows_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No training rows yet. Waiting for first reward log...</p>"

    header = (
        "<tr>"
        "<th>#</th><th>Prompt</th><th>Completion</th><th>reward_f1</th>"
        "<th>reward_format</th><th>reward_detection</th><th>reward_precision</th>"
        "<th>reward_workflow</th><th>reward_inspect_coverage</th><th>reward_coverage</th><th>Advantage</th>"
        "</tr>"
    )
    body_rows = []
    for i, row in enumerate(rows, start=1):
        body_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{escape(str(row.get('prompt', '')))}</td>"
            f"<td>{escape(str(row.get('completion', '')))}</td>"
            f"<td>{float(row.get('reward_f1', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_format', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_detection', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_precision', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_workflow', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_inspect_coverage', 0.0)):.4f}</td>"
            f"<td>{float(row.get('reward_coverage', 0.0)):.4f}</td>"
            f"<td>{float(row.get('advantage', 0.0)):+.4f}</td>"
            "</tr>"
        )

    return (
        "<table style='width:100%; border-collapse:collapse; font-size:13px;'>"
        "<thead style='background:#f3f4f6;'>"
        f"{header}"
        "</thead>"
        "<tbody>"
        f"{''.join(body_rows)}"
        "</tbody>"
        "</table>"
    )

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        s = _TRAIN_STATUS
        status_label = "Done" if s["done"] else "Training"
        table_html = _render_rows_table(s.get("rows", []))
        body = (
            "<html><head><title>Auditrix GRPO Training</title></head>"
            "<body style='font-family:Segoe UI, sans-serif; padding:20px; background:#fafafa;'>"
            "<h2 style='margin:0 0 8px 0;'>Auditrix GRPO Training</h2>"
            f"<p style='margin:4px 0;'><strong>Status:</strong> {status_label}</p>"
            f"<p style='margin:4px 0;'><strong>Step:</strong> {s['step']}/{s['total']}</p>"
            f"<p style='margin:4px 0 14px 0;'><strong>Latest reward_from_env:</strong> {float(s.get('last_reward', 0.0)):.4f}</p>"
            "<h3 style='margin:0 0 10px 0;'>Latest Batch (Structured Table)</h3>"
            f"{table_html}"
            "</body></html>"
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
from openenv_compliance_audit.rules import RULES, VIOLATION_SEVERITY

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Auditrix, an expert compliance auditor that reasons under uncertainty.

TOOL WORKFLOW:
1. inspect_record(record_id)            — reveal a record's fields (required first)
2. prioritize_rules(rule_priority_order) — declare intended rule priority (highest-severity first)
3. apply_rule(record_id, rule_id)       — test a rule; returns verdict + confidence + evidence
4. request_evidence(record_id, rule_id) — gather evidence before committing (free action)
5. flag_violation(record_id, rule_id)   — officially flag a confirmed violation
6. retract_flag(record_id, rule_id)     — undo a previous flag if new evidence changes your view
7. mark_compliant(record_id)            — declare a record has no violations
8. generate_report(report)              — submit final audit report (ends episode)
9. finish                               — end episode when report already submitted

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
Even with revealed fields, include inspect_record actions first for each record to match environment workflow.
Analyze every record against every active rule, then output ONE JSON object only.
No prose. No markdown fences. Raw JSON only.

Use the exact environment action schema and names:
    action_type in {
        inspect_record, apply_rule, request_evidence, flag_violation,
        retract_flag, mark_compliant, prioritize_rules, generate_report, finish
    }
    fields: record_id | rule_id | rule_priority_order | report

Output format:
{
    "actions": [
        {"action_type": "prioritize_rules", "rule_priority_order": ["R9", "R1", "R4"]},
        {"action_type": "flag_violation", "record_id": "E001", "rule_id": "R1"},
        {"action_type": "mark_compliant", "record_id": "E002"}
    ],
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
            f"  {r['rule_id']} [{r.get('severity', 'medium')}]: {r.get('description', '')}"
            f" | condition: {r.get('condition', '')}"
            for r in available_rules
        )
    else:
        rules_text = "\n".join(
            f"  {rid} [{VIOLATION_SEVERITY.get(rid, 'medium')}]: {RULES[rid].description}"
            f" | condition: {RULES[rid].condition_summary}"
            for rid in task.active_rule_ids
            if rid in RULES
        )

    action_schema_text = (
        "  action_type: inspect_record | apply_rule | request_evidence | flag_violation | "
        "retract_flag | mark_compliant | prioritize_rules | generate_report | finish\n"
        "  record_id: string|null\n"
        "  rule_id: string|null\n"
        "  rule_priority_order: string[]|null\n"
        "  report: object|null"
    )

    user_content = (
        f"TASK: {task_id}\n\n"
        f"ACTIVE RULES:\n{rules_text}\n\n"
        f"ACTION SCHEMA:\n{action_schema_text}\n\n"
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


def _normalize_output(output: dict) -> dict:
    """Normalize action-centric outputs into report-centric fields used by rewards.

    Supports either direct report JSON (violations/compliant_records/summary)
    or action-driven JSON containing an "actions" list with action_type entries.
    """
    if not isinstance(output, dict):
        return {"violations": [], "compliant_records": [], "summary": ""}

    if all(k in output for k in ("violations", "compliant_records", "summary")):
        return output

    actions = output.get("actions") if isinstance(output.get("actions"), list) else []
    violations = []
    compliant = set()

    for action in actions:
        if not isinstance(action, dict):
            continue
        a_type = action.get("action_type")
        if a_type == "flag_violation" and action.get("record_id") and action.get("rule_id"):
            violations.append(
                {
                    "record_id": str(action.get("record_id")),
                    "rule_id": str(action.get("rule_id")),
                    "reason": str(action.get("reason", "from action sequence")),
                }
            )
        elif a_type == "mark_compliant" and action.get("record_id"):
            compliant.add(str(action.get("record_id")))

    summary = output.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = f"Found {len(violations)} violations from action sequence."

    return {
        "violations": violations,
        "compliant_records": sorted(compliant),
        "summary": summary,
    }

_RID_KEYS  = ("record_id", "employee_id", "id", "emp_id", "employee", "record")
_RULE_KEYS = ("rule_id", "rule", "rule_name", "violation_rule",
              "violated_rule", "violated_rule_id", "ruleid")
_VALID_ACTIONS = {
    "inspect_record",
    "apply_rule",
    "request_evidence",
    "flag_violation",
    "retract_flag",
    "mark_compliant",
    "prioritize_rules",
    "generate_report",
    "finish",
}


def _extract_actions(output: dict) -> list[dict]:
    actions = output.get("actions") if isinstance(output, dict) else None
    if not isinstance(actions, list):
        return []
    return [a for a in actions if isinstance(a, dict)]


def _format_score(output: dict) -> float:
    """Strict format quality score in [0, 1]."""
    if not isinstance(output, dict):
        return 0.0

    has_actions = isinstance(output.get("actions"), list)
    has_violations = isinstance(output.get("violations"), list)
    has_compliant = isinstance(output.get("compliant_records"), list)
    has_summary = isinstance(output.get("summary"), str) and bool(output.get("summary", "").strip())

    if not (has_actions and has_violations and has_compliant and has_summary):
        return 0.0

    for v in output.get("violations", []):
        if not isinstance(v, dict):
            return 0.0
        if not all(k in v for k in ("record_id", "rule_id", "reason")):
            return 0.0
    return 1.0


def _workflow_score(output: dict, task_id: str) -> float:
    """Score action sequence quality against environment constraints in [0, 1]."""
    actions = _extract_actions(output)
    if not actions:
        return 0.0

    task = TASKS[task_id]
    valid_records = {r.record_id for r in task.records}
    valid_rules = set(task.active_rule_ids)
    inspected: set[str] = set()

    checks = 0
    violations = 0
    saw_prioritize = False

    for i, action in enumerate(actions):
        checks += 1
        action_type = action.get("action_type")
        if action_type not in _VALID_ACTIONS:
            violations += 1
            continue

        if action_type == "prioritize_rules":
            saw_prioritize = True
            order = action.get("rule_priority_order")
            if not isinstance(order, list) or not order:
                violations += 1
            elif any((not isinstance(rid, str)) or (rid not in valid_rules) for rid in order):
                violations += 1
            if i != 0:
                violations += 1
            continue

        needs_record = action_type in {
            "inspect_record", "apply_rule", "request_evidence",
            "flag_violation", "retract_flag", "mark_compliant"
        }
        record_id = action.get("record_id")
        if needs_record:
            if not isinstance(record_id, str) or record_id not in valid_records:
                violations += 1
                continue

        if action_type == "inspect_record":
            inspected.add(record_id)
            continue

        if action_type in {"apply_rule", "request_evidence", "flag_violation", "retract_flag"}:
            rule_id = action.get("rule_id")
            if not isinstance(rule_id, str) or rule_id not in valid_rules:
                violations += 1
            if record_id not in inspected:
                violations += 1
            continue

        if action_type == "mark_compliant":
            if record_id not in inspected:
                violations += 1
            continue

        if action_type in {"generate_report", "finish"}:
            if len(inspected) < len(valid_records):
                violations += 1
            continue

    score = max(0.0, 1.0 - (violations / max(checks, 1)))
    if saw_prioritize:
        score = min(1.0, score + 0.1)
    return score

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
    output = _normalize_output(output)
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
            "task_id"   : tid,
            "parsed"    : parsed,
            "detection" : round(det, 3),
            "precision" : round(prec, 3),
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
        scores.append(0.1 * _format_score(out))
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


def reward_workflow(completions, prompts, task_id, **kwargs) -> list[float]:
    scores = []
    for completion, tid in zip(completions, task_id):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        workflow = _workflow_score(_parse_output(text), tid)
        scores.append(float(workflow) * 0.25)
    return scores


def _inspect_coverage_score(output: dict, task_id: str) -> float:
    """Score inspection coverage in [0, 1], strict before terminal actions.

    If generate_report/finish is present, all records should be inspected at least once.
    """
    actions = _extract_actions(output)
    if not actions:
        return 0.0

    task = TASKS[task_id]
    valid_records = {r.record_id for r in task.records}
    if not valid_records:
        return 1.0

    inspected = {
        a.get("record_id")
        for a in actions
        if a.get("action_type") == "inspect_record" and isinstance(a.get("record_id"), str)
    }
    inspected = {rid for rid in inspected if rid in valid_records}

    coverage = len(inspected) / len(valid_records)
    has_terminal = any(a.get("action_type") in {"generate_report", "finish"} for a in actions)

    if has_terminal:
        return 1.0 if coverage >= 1.0 else 0.0
    return coverage


def reward_inspect_coverage(completions, prompts, task_id, **kwargs) -> list[float]:
    scores = []
    for completion, tid in zip(completions, task_id):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = _inspect_coverage_score(_parse_output(text), tid)
        scores.append(float(score) * 0.20)
    return scores

# ── Clean logging callback ────────────────────────────────────────────────────
class AuditLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step    = state.global_step
        reward  = logs.get("reward")
        _TRAIN_STATUS["step"] = step
        _TRAIN_STATUS["last_reward"] = float(reward) if reward is not None else float(_TRAIN_STATUS.get("last_reward", 0.0))

        if reward is None: return  # skip non-reward log lines

        print(f"\n{'═'*72}")
        print(f"Step {step:>4}/{args.max_steps}  |  reward_from_env={reward:.4f}")
        print(f"{'─'*72}")
        print("| # | reward_f1 | reward_format | reward_detection | reward_precision | reward_workflow | reward_inspect_coverage | reward_coverage | Advantage |")
        print("|---|-----------|---------------|------------------|------------------|-----------------|-------------------------|-----------------|-----------|")

        batch_mean = sum(e["f1"] for e in _LAST_BATCH) / len(_LAST_BATCH) if _LAST_BATCH else 0.0
        rows = []

        for i, entry in enumerate(_LAST_BATCH):
            advantage  = entry["f1"] - batch_mean
            parsed = entry.get("parsed") or {}
            reward_format = 0.1 * _format_score(parsed)
            reward_detection = float(entry.get("detection", 0.0)) * 0.6
            reward_precision = float(entry.get("precision", 0.0)) * 0.4
            reward_workflow = float(_workflow_score(parsed, entry.get("task_id", TRAIN_TASKS[0]))) * 0.25
            reward_inspect_coverage = float(_inspect_coverage_score(parsed, entry.get("task_id", TRAIN_TASKS[0]))) * 0.20

            rows.append(
                {
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "reward_f1": entry["f1"],
                    "reward_format": reward_format,
                    "reward_detection": reward_detection,
                    "reward_precision": reward_precision,
                    "reward_workflow": reward_workflow,
                    "reward_inspect_coverage": reward_inspect_coverage,
                    "reward_coverage": entry["coverage"],
                    "advantage": advantage,
                }
            )

            print(
                f"| {i+1} | {entry['f1']:.4f} | {reward_format:.4f} | {reward_detection:.4f} | "
                f"{reward_precision:.4f} | {reward_workflow:.4f} | {reward_inspect_coverage:.4f} | {entry['coverage']:.4f} | {advantage:+.4f} |"
            )

            print(f"  [{i+1}] Prompt    : {entry['prompt']}")
            print(f"       Completion: {entry['completion']}")
            print(f"       reward_f1={entry['f1']:.4f} | "
                  f"reward_format={reward_format:.1f} | "
                f"reward_detection={reward_detection:.4f} | "
                f"reward_precision={reward_precision:.4f} | "
                f"reward_workflow={reward_workflow:.4f} | "
                f"reward_inspect_coverage={reward_inspect_coverage:.4f} | "
                  f"reward_coverage={entry['coverage']:.4f} | "
                  f"Advantage={advantage:+.4f}")
            print()

        _TRAIN_STATUS["rows"] = rows

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
        reward_funcs=[
            reward_f1,
            reward_format,
            reward_detection,
            reward_precision,
            reward_workflow,
            reward_inspect_coverage,
        ],
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