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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            :root {
                --bg-primary: #f5f7fa;
                --bg-secondary: #ffffff;
                --bg-tertiary: #fbfcfe;
                --text-primary: #202124;
                --text-secondary: #5f6368;
                --border-color: #e6e8eb;
                --border-light: #e8eaed;
                --border-lighter: #eceff3;
                --link-color: #1a73e8;
                --link-bg-hover: #eef4ff;
                --link-border: #d2e3fc;
                --header-border: #eceff3;
                --table-header-bg: #f8f9fb;
                --table-row-hover: #f7faff;
                --table-accent: #eef4ff;
                --table-accent-text: #123a70;
                --code-bg: #f1f3f4;
            }
            body.dark-mode {
                --bg-primary: #1a1a1a;
                --bg-secondary: #242424;
                --bg-tertiary: #2d2d2d;
                --text-primary: #e8e8e8;
                --text-secondary: #b0b0b0;
                --border-color: #3a3a3a;
                --border-light: #404040;
                --border-lighter: #454545;
                --link-color: #6bb3ff;
                --link-bg-hover: #1e3a52;
                --link-border: #2a4a6a;
                --header-border: #333333;
                --table-header-bg: #2d2d2d;
                --table-row-hover: #2a2a2a;
                --table-accent: #1e3a52;
                --table-accent-text: #7fb3e5;
                --code-bg: #2d2d2d;
            }
            body {
                font-family: "Google Sans", "Roboto", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                padding: 28px 16px;
                transition: background-color 0.3s, color 0.3s;
            }
            .container {
                max-width: 1120px;
                margin: 0 auto;
                background: var(--bg-secondary);
                border-radius: 14px;
                box-shadow: 0 8px 28px rgba(60, 64, 67, 0.12);
                overflow: hidden;
                border: 1px solid var(--border-color);
            }
            .header {
                background: var(--bg-secondary);
                color: var(--text-primary);
                padding: 30px 34px 22px 34px;
                border-bottom: 1px solid var(--header-border);
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
            }
            .header-content {
                flex: 1;
            }
            .header h1 {
                font-size: 1.9rem;
                font-weight: 700;
                letter-spacing: 0.2px;
                margin-bottom: 8px;
            }
            .header p {
                font-size: 0.98rem;
                color: var(--text-secondary);
                max-width: 880px;
            }
            .theme-toggle {
                width: 40px;
                height: 20px;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-light);
                border-radius: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                padding: 2px;
                transition: background-color 0.3s, border-color 0.3s;
                position: relative;
            }
            .theme-toggle.dark-mode-active {
                justify-content: flex-end;
            }
            .theme-toggle::after {
                content: '';
                width: 18px;
                height: 18px;
                background: var(--bg-secondary);
                border-radius: 18px;
                transition: all 0.3s;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .theme-toggle:hover {
                border-color: var(--link-border);
            }
            .content {
                padding: 26px 34px 34px 34px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 14px;
                margin-bottom: 28px;
            }
            .stat-card {
                background: var(--bg-tertiary);
                padding: 16px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid var(--border-lighter);
            }
            .stat-number { font-size: 1.72rem; font-weight: 700; color: var(--link-color); }
            .stat-label { color: var(--text-secondary); margin-top: 4px; font-size: 0.92rem; }
            .section {
                margin-bottom: 28px;
            }
            .section h2 {
                color: var(--text-primary);
                margin-bottom: 14px;
                font-size: 1.1rem;
                font-weight: 600;
                padding-bottom: 8px;
                border-bottom: 1px solid var(--border-light);
            }
            .section p {
                color: var(--text-secondary);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 0.94rem;
            }
            th, td {
                padding: 11px 10px;
                text-align: left;
                border-bottom: 1px solid var(--border-lighter);
                color: var(--text-primary);
            }
            th {
                background: var(--table-header-bg);
                font-weight: 600;
                color: var(--text-primary);
            }
            tr:hover { background: var(--table-row-hover); }
            .average-row {
                background: var(--table-accent);
                font-weight: bold;
            }
            .average-row td {
                color: var(--table-accent-text) !important;
            }
            .api-link {
                display: inline-block;
                margin: 5px 6px 0 0;
                padding: 7px 14px;
                background: var(--bg-secondary);
                color: var(--link-color);
                text-decoration: none;
                border: 1px solid var(--link-border);
                border-radius: 999px;
                font-size: 0.9em;
            }
            .api-link:hover {
                background: var(--link-bg-hover);
                text-decoration: none;
            }
            .footer {
                background: var(--table-header-bg);
                padding: 16px;
                text-align: center;
                color: var(--text-secondary);
                border-top: 1px solid var(--border-light);
            }
            code {
                background: var(--code-bg);
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
                font-size: 0.9em;
                color: var(--text-primary);
            }
            pre {
                background: var(--table-header-bg) !important;
                color: var(--text-primary);
                border: 1px solid var(--border-light);
            }
            a[style*="color"] {
                color: var(--link-color) !important;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-content">
                    <h1>OpenEnv Compliance Audit</h1>
                    <p>Interactive Environment for Evaluating AI Agents on Compliance Audit Tasks</p>
                </div>
                <button class="theme-toggle" id="themeToggle" title="Toggle dark mode"></button>
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
                    <h2>Available Tasks</h2>
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
                    <h2>API Endpoints</h2>
                    <p>Access the environment programmatically via these REST endpoints:</p>
                    <div style="margin-top: 15px;">
                        <a href="/health" class="api-link">GET /health</a>
                        <a href="/tasks" class="api-link">GET /tasks</a>
                        <a href="/baseline" class="api-link">GET /baseline</a>
                        <a href="/docs" class="api-link">Interactive API Docs</a>
                    </div>
                    <p style="margin-top: 20px; font-size: 0.9em; color: var(--text-secondary);">
                        POST requests: <code>/reset</code>, <code>/step</code>, <code>/state</code>
                    </p>
                </div>
                
                <div class="section">
                    <h2>Baseline Scores</h2>
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
                        <tr class="average-row">
                            <td>Average</td>
                            <td><strong>0.7217</strong></td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Quick Start</h2>
                    <p>Start an episode and take actions:</p>
                    <pre style="padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 0.85em;">
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
                <p>OpenEnv Compliance Audit | <a href="https://github.com/Kowshikv07/Auditrix" style="color: var(--link-color);">GitHub</a> | OpenEnv Framework</p>
            </div>
        </div>
        <script>
            const themeToggle = document.getElementById('themeToggle');
            const body = document.body;
            const savedTheme = localStorage.getItem('theme');
            
            if (savedTheme === 'dark') {
                body.classList.add('dark-mode');
                themeToggle.classList.add('dark-mode-active');
            }
            
            themeToggle.addEventListener('click', () => {
                body.classList.toggle('dark-mode');
                themeToggle.classList.toggle('dark-mode-active');
                const isDarkMode = body.classList.contains('dark-mode');
                localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
            });
        </script>
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
