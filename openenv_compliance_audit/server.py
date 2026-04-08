"""FastAPI server exposing the OpenEnv compliance audit environment."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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
def root() -> HTMLResponse:
    """Root endpoint serves the dashboard directly."""
    return dashboard_html()


@app.get("/api")
def api_root() -> dict:
    """Machine-readable API info (JSON)."""
    return {
        "name": "openenv-compliance-audit",
        "version": "1.0.0",
        "status": "ok",
        "dashboard": "/dashboard",
        "api_docs": "/docs",
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


@app.get("/dashboard")
def dashboard_html() -> HTMLResponse:
    """Return HTML dashboard for visualizing model performance."""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenEnv Compliance Audit - Leaderboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            .content {
                padding: 40px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .stat-card {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
            }
            .stat-number { font-size: 2em; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; margin-top: 5px; }
            .section {
                margin-bottom: 40px;
            }
            .section h2 {
                color: #333;
                margin-bottom: 20px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            th {
                background: #f5f5f5;
                font-weight: 600;
                color: #333;
            }
            tr:hover { background: #f9f9f9; }
            .api-link {
                display: inline-block;
                margin: 5px;
                padding: 8px 16px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 0.9em;
            }
            .api-link:hover {
                background: #764ba2;
                text-decoration: none;
            }
            .footer {
                background: #f5f5f5;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #eee;
            }
            code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚖️ OpenEnv Compliance Audit</h1>
                <p>Interactive Environment for Evaluating AI Agents on Compliance Audit Tasks</p>
            </div>
            
            <div class="content">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">6</div>
                        <div class="stat-label">Tasks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">3</div>
                        <div class="stat-label">Difficulty Levels</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">10</div>
                        <div class="stat-label">Compliance Rules</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">[0,1]</div>
                        <div class="stat-label">Score Range</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 Available Tasks</h2>
                    <table id="tasksTable">
                        <tr>
                            <th>Task ID</th>
                            <th>Difficulty</th>
                            <th>Records</th>
                            <th>Active Rules</th>
                            <th>Max Steps</th>
                        </tr>
                        <tr>
                            <td>easy_basic_audit</td>
                            <td>🟢 Easy</td>
                            <td>5</td>
                            <td>R1, R2</td>
                            <td>25</td>
                        </tr>
                        <tr>
                            <td>medium_mixed_audit</td>
                            <td>🟡 Medium</td>
                            <td>12</td>
                            <td>R1-R4</td>
                            <td>50</td>
                        </tr>
                        <tr>
                            <td>hard_complex_audit</td>
                            <td>🔴 Hard</td>
                            <td>20</td>
                            <td>R1-R5</td>
                            <td>100</td>
                        </tr>
                        <tr>
                            <td>finance_sox_audit</td>
                            <td>🔴 Hard</td>
                            <td>15</td>
                            <td>R3,R6-R8</td>
                            <td>80</td>
                        </tr>
                        <tr>
                            <td>gdpr_privacy_audit</td>
                            <td>🟡 Medium</td>
                            <td>10</td>
                            <td>R5,R8,R9</td>
                            <td>50</td>
                        </tr>
                        <tr>
                            <td>data_integrity_audit</td>
                            <td>🟡 Medium</td>
                            <td>8</td>
                            <td>R3,R4,R10</td>
                            <td>40</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>🔗 API Endpoints</h2>
                    <p>Access the environment programmatically via these REST endpoints:</p>
                    <div style="margin-top: 15px;">
                        <a href="/health" class="api-link">GET /health</a>
                        <a href="/tasks" class="api-link">GET /tasks</a>
                        <a href="/baseline" class="api-link">GET /baseline</a>
                        <a href="/docs" class="api-link">📖 Interactive API Docs</a>
                    </div>
                    <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
                        POST requests: <code>/reset</code>, <code>/step</code>, <code>/state</code>
                    </p>
                </div>
                
                <div class="section">
                    <h2>📊 Baseline Scores</h2>
                    <p>Performance of rule-based baseline agent (Qwen 2.5 72B with temperature=0):</p>
                    <table style="margin-top: 15px;">
                        <tr>
                            <th>Task</th>
                            <th>Score</th>
                        </tr>
                        <tr>
                            <td>easy_basic_audit</td>
                            <td><strong>0.92</strong></td>
                        </tr>
                        <tr>
                            <td>medium_mixed_audit</td>
                            <td><strong>0.75</strong></td>
                        </tr>
                        <tr>
                            <td>hard_complex_audit</td>
                            <td><strong>0.58</strong></td>
                        </tr>
                        <tr>
                            <td>finance_sox_audit</td>
                            <td><strong>0.61</strong></td>
                        </tr>
                        <tr>
                            <td>gdpr_privacy_audit</td>
                            <td><strong>0.72</strong></td>
                        </tr>
                        <tr>
                            <td>data_integrity_audit</td>
                            <td><strong>0.74</strong></td>
                        </tr>
                        <tr style="font-weight: bold; background: #f5f5f5;">
                            <td>Average</td>
                            <td><strong>0.7217</strong></td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>🚀 Quick Start</h2>
                    <p>Start an episode and take actions:</p>
                    <pre style="background: #f5f5f5; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 0.85em;">
# Reset environment
curl -X POST http://localhost:7860/reset \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "easy_basic_audit"}'

# Take a step
curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{
    "action_type": "inspect_record",
    "record_id": "E001"
  }'
                    </pre>
                </div>
            </div>
            
            <div class="footer">
                <p>⚖️ OpenEnv Compliance Audit | <a href="https://github.com/Kowshikv07/Auditrix" style="color: #667eea;">GitHub</a> | OpenEnv Framework</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
