from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from .environment import TicketTriageEnv
from .models import TicketTriageAction
from .tasks import TASKS


class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None)


env = TicketTriageEnv()
app = FastAPI(title="OpenEnv Ticket Triage", version="0.1.0")


@app.get("/")
def root() -> dict:
    return {
        "name": "openenv-ticket-triage",
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
            }
            for t in TASKS.values()
        ],
    }


@app.get("/reset")
def reset_get(task_id: str | None = None) -> dict:
    observation = env.reset(task_id=task_id)
    return {"observation": observation.model_dump()}


@app.post("/reset")
def reset_post(payload: ResetRequest) -> dict:
    observation = env.reset(task_id=payload.task_id)
    return {"observation": observation.model_dump()}


@app.post("/step")
def step(action: TicketTriageAction) -> dict:
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return env.state().model_dump()


def run_server() -> None:
    uvicorn.run("openenv_ticket_triage.server:app", host="0.0.0.0", port=7860)
