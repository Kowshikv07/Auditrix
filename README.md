# OpenEnv Ticket Triage

A deterministic OpenEnv-compatible environment that simulates **real customer support operations**: triaging inbound tickets, setting priority, routing to specialized teams, escalating compliance incidents, and resolving cases safely.

## Why This Environment Matters

Customer support triage is a genuine high-impact human workflow. Teams do this every day to prevent SLA misses, security incidents, and compliance failures. This environment models that workflow in a way that is:

- realistic (priority/routing/compliance tradeoffs)
- deterministic (reproducible grading)
- useful for agent evaluation and training

## Environment Summary

- Benchmark name: `openenv_ticket_triage`
- Interface: `reset()`, `step(action)`, `state()`
- Episode style: finite-horizon, task-oriented
- Reward: shaped per step (partial progress + penalties) and bounded in `[0.0, 1.0]`

## Action Space

Action model: `TicketTriageAction`

Fields:

- `action_type`: one of
  - `inspect_ticket`
  - `set_priority`
  - `assign_team`
  - `request_customer_reply`
  - `add_internal_note`
  - `escalate_compliance`
  - `resolve_ticket`
  - `finish`
- `ticket_id`: target ticket identifier (required for all task actions except `finish`)
- `value`: optional action payload
  - `set_priority`: `low|medium|high|critical`
  - `assign_team`: `billing|support|product|security|fraud|privacy`
  - `add_internal_note`: free text

## Observation Space

Observation model: `TicketTriageObservation`

Fields:

- `task_id`, `task_title`, `objective`
- `step_index`, `max_steps`
- `progress_score` in `[0.0, 1.0]`
- `visible_tickets[]` where each item includes:
  - `ticket_id`, `subject`, `customer_tier`, `age_hours`
  - `inspected`, `priority`, `assigned_team`
  - `requested_customer_reply`, `compliance_escalated`, `resolved`
- `action_history[]`
- `last_action_error`

## Reward Design

The reward function is shaped to provide trajectory-level learning signal:

- positive partial credit for valid progress actions:
  - new inspection
  - correct priority assignment
  - correct team routing
  - required customer follow-up request
  - required compliance escalation
  - correct resolution
- small reward for documentation (`add_internal_note`)
- penalties for undesirable behavior:
  - repeated actions
  - incorrect classification/routing
  - unnecessary escalation
  - invalid actions

Reward is clipped to `[0.0, 1.0]` on every step. Final score from deterministic grader contributes a terminal bonus on `finish`.

## Tasks and Difficulty

### 1) Easy: `easy_refund_priority`

Objective:

- Correctly triage and resolve one urgent refund incident.

Why easy:

- single ticket, clear routing to billing, straightforward urgency.

### 2) Medium: `medium_mixed_queue`

Objective:

- Triage 3 tickets with different intents (outage, feature request, phishing report).
- Request customer follow-up for phishing evidence.
- Resolve at least the resolvable subset.

Why medium:

- multi-ticket queue with mixed urgency and team ownership.

### 3) Hard: `hard_regulated_incident`

Objective:

- Handle a regulated queue that includes payment-data exposure and privacy deletion requests.
- Correctly escalate compliance-required cases.
- Route, prioritize, and resolve what is resolvable without unsafe actions.

Why hard:

- requires cross-ticket consistency, compliance judgment, and efficiency.

## Deterministic Graders (Agent Graders)

Three deterministic grader agents are implemented:

- `EasyGraderAgent`
- `MediumGraderAgent`
- `HardGraderAgent`

Each grader computes weighted deterministic components in `[0,1]`:

- priority accuracy
- team accuracy
- resolution coverage
- compliance accuracy (hard)
- customer reply accuracy (medium/hard)
- efficiency (step usage + penalty profile)

All task scores are normalized to `[0.0, 1.0]`.

## Project Structure

- `openenv_ticket_triage/models.py`: typed Pydantic models
- `openenv_ticket_triage/tasks.py`: task definitions
- `openenv_ticket_triage/graders.py`: deterministic grader agents
- `openenv_ticket_triage/environment.py`: environment logic (`reset`, `step`, `state`)
- `openenv_ticket_triage/server.py`: FastAPI endpoints for deployment
- `openenv.yaml`: OpenEnv metadata manifest
- `inference.py`: baseline runner using OpenAI client
- `Dockerfile`: container image for local + HF Spaces deployment

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Run locally:

```bash
uvicorn openenv_ticket_triage.server:app --host 0.0.0.0 --port 7860
```

API smoke test:

```bash
curl http://localhost:7860/reset
```

## Baseline Inference

The required script is at repository root: `inference.py`.

Environment variables:

- `OPENAI_API_KEY` (preferred) or `HF_TOKEN`
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)

Run:

```bash
python inference.py
```

The script emits required structured stdout lines:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

And prints final per-task and average score JSON summary.

## Baseline Reproducibility

Reproducibility controls:

- fixed random seed (`SEED=7`)
- deterministic sampling (`temperature=0`)
- deterministic task fixtures and graders

## Docker

Build:

```bash
docker build -t openenv-ticket-triage .
```

Run:

```bash
docker run --rm -p 7860:7860 openenv-ticket-triage
```

## Hugging Face Spaces Deployment

Recommended settings:

- Space SDK: `Docker`
- Add tag: `openenv`
- Exposed port: `7860`
- Health endpoint: `/health`

The app responds to both `GET /reset` and `POST /reset` for validator compatibility.

## Validation Checklist

- [x] Real-world task simulation (customer support triage)
- [x] Typed models for Observation, Action, Reward
- [x] Full `reset()`, `step()`, `state()` implementation
- [x] 3 tasks (easy, medium, hard) with deterministic graders
- [x] Shaped reward with partial progress and penalties
- [x] `inference.py` baseline using OpenAI client
- [x] `openenv.yaml`
- [x] Dockerfile for containerized execution

## Example Deterministic Reference Policy Scores

A deterministic handcrafted policy (not LLM) should achieve roughly:

- Easy: 0.95 - 1.00
- Medium: 0.75 - 0.92
- Hard: 0.62 - 0.86

Exact LLM baseline results vary by model but are reproducible given the same endpoint/model and deterministic inference settings.
