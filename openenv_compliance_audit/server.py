"""FastAPI server exposing the OpenEnv compliance audit environment."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .environment import ComplianceAuditEnv
from .models import AuditAction
from .tasks import TASKS

import uvicorn


class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None)


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
    return {
        "benchmark": env.benchmark_name,
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
def reset_get(task_id: str | None = None) -> dict:
    observation = env.reset(task_id=task_id)
    return {"observation": observation.model_dump()}


@app.post("/reset")
async def reset_post(request: Request, task_id: str | None = None) -> dict:
    body_task_id: str | None = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            body_task_id = body.get("task_id")
    except Exception:
        # Accept empty or non-JSON POST bodies and fall back to query/default.
        body_task_id = None

    observation = env.reset(task_id=body_task_id or task_id)
    return {"observation": observation.model_dump()}


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
