"""FastAPI server exposing the OpenEnv compliance audit environment."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .environment import ComplianceAuditEnv
from .models import AuditAction
from .rules import RULES
from .tasks import TASKS

import uvicorn


class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None)
    seed: int | None = Field(default=None)


env = ComplianceAuditEnv()
app = FastAPI(
    title="OpenEnv Compliance Audit",
    version="1.0.0",
    description=(
        "An AI agent environment for performing iterative compliance audits "
        "on structured organisational records using predefined rules."
    ),
)


@app.get("/")
def root() -> dict:
    return {
        "name": "openenv-compliance-audit",
        "version": "1.0.0",
        "status": "ok",
        "tasks": list(TASKS.keys()),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> dict:
    action_schema = {
        "action_type": [
            "inspect_record",
            "apply_rule",
            "flag_violation",
            "mark_compliant",
            "generate_report",
            "finish",
        ],
        "record_id": "string|null",
        "rule_id": "string|null",
        "report": "object|null",
    }

    return {
        "benchmark": env.benchmark_name,
        "score_range": [0.0, 1.0],
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "tasks": {"method": "GET", "path": "/tasks"},
            "reset": {"method": "POST", "path": "/reset"},
            "step": {"method": "POST", "path": "/step"},
            "state": {"method": "GET", "path": "/state"},
            "baseline": {"method": "GET", "path": "/baseline"},
        },
        "reset_request_schema": {
            "task_id": "string|null",
            "seed": "integer|null",
        },
        "action_schema": action_schema,
        "tasks": [
            {
                "task_id": t.task_id,
                "title": t.title,
                "difficulty": t.difficulty,
                "objective": t.objective,
                "max_steps": t.max_steps,
                "active_rules": t.active_rule_ids,
                "num_records": len(t.records),
            }
            for t in TASKS.values()
        ],
    }


@app.get("/reset")
def reset_get(task_id: str | None = None, seed: int | None = None) -> dict:
    observation = env.reset(task_id=task_id, seed=seed)
    return {
        "observation": observation.model_dump(),
        "meta": {"task_id": observation.task_id, "seed": seed},
    }


@app.post("/reset")
async def reset_post(request: Request, task_id: str | None = None) -> dict:
    body_task_id: str | None = None
    body_seed: int | None = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            body_task_id = body.get("task_id")
            body_seed = body.get("seed")
    except Exception:
        # Accept empty or non-JSON POST bodies and fall back to query/default.
        body_task_id = None
        body_seed = None

    resolved_seed = body_seed
    observation = env.reset(task_id=body_task_id or task_id, seed=resolved_seed)
    return {
        "observation": observation.model_dump(),
        "meta": {"task_id": observation.task_id, "seed": resolved_seed},
    }


def _run_rule_based_baseline(task_id: str) -> Dict[str, Any]:
    baseline_env = ComplianceAuditEnv(task_id=task_id)
    baseline_env.reset(task_id=task_id)
    task = TASKS[task_id]
    all_fields = [record.fields for record in task.records]

    step_count = 0

    for record in task.records:
        baseline_env.step(AuditAction(action_type="inspect_record", record_id=record.record_id))
        step_count += 1

    for record in task.records:
        for rule_id in task.active_rule_ids:
            if RULES[rule_id].evaluate(record.fields, all_fields):
                baseline_env.step(
                    AuditAction(
                        action_type="flag_violation",
                        record_id=record.record_id,
                        rule_id=rule_id,
                    )
                )
                step_count += 1

    for record in task.records:
        if not record.expected_violations:
            baseline_env.step(AuditAction(action_type="mark_compliant", record_id=record.record_id))
            step_count += 1

    final = baseline_env.step(
        AuditAction(
            action_type="generate_report",
            report={
                "summary": "Deterministic rule-based baseline report.",
                "flagged_violations": [],
                "compliant_records": [],
                "recommendations": ["Review low-confidence flags manually."],
            },
        )
    )
    step_count += 1

    return {
        "task_id": task_id,
        "score": float(final.info.get("task_score", 0.0)),
        "steps_used": step_count,
        "max_steps": task.max_steps,
    }


@app.get("/baseline")
def baseline() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for task_id in TASKS:
        results.append(_run_rule_based_baseline(task_id))

    mean_score = sum(item["score"] for item in results) / max(len(results), 1)
    return {
        "benchmark": env.benchmark_name,
        "agent": "rule_based",
        "score_range": [0.0, 1.0],
        "mean_score": round(mean_score, 4),
        "tasks": results,
    }


@app.post("/step")
def step(action: AuditAction) -> dict:
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        # Return a client error instead of an internal server error when state
        # is requested before a task is initialized.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run_server() -> None:
    uvicorn.run(
        "openenv_compliance_audit.server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


def main() -> None:
    """Console entrypoint expected by OpenEnv validators."""
    run_server()
