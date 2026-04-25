"""FastAPI server exposing the OpenEnv compliance audit environment."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
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
            "request_evidence",
            "flag_violation",
            "retract_flag",
            "mark_compliant",
            "prioritize_rules",
            "generate_report",
            "finish",
        ],
        "record_id": "string|null",
        "rule_id": "string|null",
        "rule_priority_order": "string[]|null",
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
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    logs_file = Path(__file__).resolve().parent.parent / "model-benchmark-logs" / "inference_runs.jsonl"
    model_rows = ""

    if logs_file.exists():
        with open(logs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rewards = item.get("rewards", [])
                if isinstance(rewards, list):
                    rewards_preview = ", ".join(str(value) for value in rewards[:2])
                    if len(rewards) > 2:
                        rewards_preview += ", ..."
                    rewards_full = ", ".join(str(value) for value in rewards)
                else:
                    rewards_preview = str(rewards)
                    rewards_full = str(rewards)

                failure_mode = item.get("failure_mode", "—")
                seed_val = item.get("seed", "—")

                model_rows += (
                    "<tr class=\"mc-row\">"
                    f"<td>{escape(str(item.get('timestamp', '')))}</td>"
                    f"<td>{escape(str(item.get('task_id', '')))}</td>"
                    f"<td>{escape(str(item.get('model', '')))}</td>"
                    f"<td>{escape(str(seed_val))}</td>"
                    f"<td>{escape(str(item.get('score', '')))}</td>"
                    f"<td>{escape(str(item.get('steps', '')))}</td>"
                    f"<td>{escape(str(item.get('success', '')))}</td>"
                    f"<td><span class=\"fm-badge fm-{escape(str(failure_mode))}\">{escape(str(failure_mode))}</span></td>"
                    "<td class=\"rewards-cell\">"
                    f"<span class=\"rewards-preview\">{escape(rewards_preview)}</span>"
                    f"<span class=\"rewards-full\">{escape(rewards_full)}</span>"
                    "</td>"
                    "</tr>"
                )

    if not model_rows:
        model_rows = "<tr><td colspan=\"9\">No benchmark records found. Run <code>python inference.py</code> to generate data.</td></tr>"

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Auditrix — OpenEnv Compliance Audit</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="description" content="Auditrix — AI agent benchmark for compliance auditing with dynamic incident mechanics">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
            :root {
                --bg: #f0f2f5;
                --surface: #ffffff;
                --surface2: #f8f9fc;
                --border: #e2e5ea;
                --text: #1a1d23;
                --text2: #5c6370;
                --accent: #4361ee;
                --accent-light: #eef1fd;
                --accent-border: #c5cef8;
                --green: #22c55e;
                --yellow: #f59e0b;
                --red: #ef4444;
                --purple: #8b5cf6;
                --orange: #f97316;
                --radius: 12px;
                --shadow: 0 4px 24px rgba(0,0,0,0.07);
            }
            body.dark {
                --bg: #0f1117;
                --surface: #1a1d27;
                --surface2: #22263a;
                --border: #2e3347;
                --text: #e8eaf0;
                --text2: #8892a4;
                --accent: #6c8eff;
                --accent-light: #1a2240;
                --accent-border: #2e4180;
                --green: #4ade80;
                --yellow: #fbbf24;
                --red: #f87171;
                --purple: #a78bfa;
                --orange: #fb923c;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                padding: 24px 16px 40px;
                transition: background .3s, color .3s;
                font-size: 14px;
            }
            .page { max-width: 1360px; margin: 0 auto; }

            /* ── Header ── */
            .header {
                background: var(--surface);
                border-radius: var(--radius);
                padding: 28px 32px;
                margin-bottom: 20px;
                box-shadow: var(--shadow);
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                border: 1px solid var(--border);
            }
            .header h1 {
                font-size: 1.65rem;
                font-weight: 700;
                letter-spacing: -.4px;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .header p { color: var(--text2); font-size: 0.9rem; max-width: 720px; }
            .feature-badge {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                background: linear-gradient(135deg, #4361ee, #7c3aed);
                color: #fff;
                font-size: 0.72rem;
                font-weight: 700;
                padding: 3px 9px;
                border-radius: 999px;
                letter-spacing: .4px;
            }
            .header-right { display: flex; align-items: center; gap: 12px; }
            .theme-btn {
                width: 44px; height: 24px;
                background: var(--surface2);
                border: 1px solid var(--border);
                border-radius: 999px;
                cursor: pointer;
                position: relative;
                transition: background .3s;
                flex-shrink: 0;
            }
            .theme-btn::after {
                content: '';
                position: absolute;
                top: 3px; left: 3px;
                width: 16px; height: 16px;
                border-radius: 50%;
                background: var(--text2);
                transition: transform .3s;
            }
            body.dark .theme-btn::after { transform: translateX(20px); background: var(--accent); }

            /* ── Stat grid ── */
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 14px;
                margin-bottom: 20px;
            }
            .stat {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px 20px;
                text-align: center;
                box-shadow: var(--shadow);
                transition: transform .15s;
            }
            .stat:hover { transform: translateY(-2px); }
            .stat-n { font-size: 2rem; font-weight: 700; color: var(--accent); line-height: 1; }
            .stat-l { color: var(--text2); font-size: 0.82rem; margin-top: 6px; }

            /* ── Feature badges ── */
            .features {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 16px 20px;
                margin-bottom: 20px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
                box-shadow: var(--shadow);
            }
            .feat-label { color: var(--text2); font-size: 0.8rem; font-weight: 600; margin-right: 4px; }
            .feat {
                display: inline-flex;
                align-items: center;
                gap: 5px;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 500;
                border: 1px solid var(--border);
                background: var(--surface2);
                color: var(--text);
                white-space: nowrap;
            }
            .feat.new { background: var(--accent-light); border-color: var(--accent-border); color: var(--accent); }

            /* ── Sections ── */
            .section {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 22px 26px;
                margin-bottom: 18px;
                box-shadow: var(--shadow);
            }
            .section h2 {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 14px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* ── Tables ── */
            table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
            th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
            th { background: var(--surface2); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .4px; color: var(--text2); }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: var(--surface2); }
            .avg-row td { font-weight: 700; color: var(--accent); background: var(--accent-light) !important; }

            /* ── Difficulty pills ── */
            .pill {
                display: inline-block;
                padding: 2px 9px;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 600;
            }
            .pill-easy   { background: #dcfce7; color: #16a34a; }
            .pill-medium { background: #fef9c3; color: #b45309; }
            .pill-hard   { background: #fee2e2; color: #dc2626; }
            .pill-extreme { background: #f3e8ff; color: #7c3aed; }
            body.dark .pill-easy   { background: #14532d; color: #86efac; }
            body.dark .pill-medium { background: #451a03; color: #fcd34d; }
            body.dark .pill-hard   { background: #450a0a; color: #fca5a5; }
            body.dark .pill-extreme { background: #2e1065; color: #c4b5fd; }

            /* ── Event type badges ── */
            .ev { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin: 1px; }
            .ev-policy  { background: #ffe4cc; color: #c2410c; }
            .ev-outage  { background: #fce7f3; color: #9d174d; }
            .ev-amend   { background: #dbeafe; color: #1e40af; }
            body.dark .ev-policy  { background: #431407; color: #fdba74; }
            body.dark .ev-outage  { background: #500724; color: #f9a8d4; }
            body.dark .ev-amend   { background: #1e3a5f; color: #93c5fd; }

            /* ── Failure mode badges ── */
            .fm-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
            .fm-none               { background: #dcfce7; color: #16a34a; }
            .fm-false_positive     { background: #fee2e2; color: #dc2626; }
            .fm-missed_violation   { background: #fef9c3; color: #b45309; }
            .fm-low_coverage       { background: #ffe4cc; color: #c2410c; }
            .fm-inefficiency       { background: #e0f2fe; color: #0369a1; }
            .fm-loop_exploit       { background: #f3e8ff; color: #7c3aed; }
            .fm-report_inconsistency { background: #fce7f3; color: #9d174d; }
            body.dark .fm-none               { background: #14532d; color: #86efac; }
            body.dark .fm-false_positive     { background: #450a0a; color: #fca5a5; }
            body.dark .fm-missed_violation   { background: #451a03; color: #fcd34d; }

            /* ── Model comparison table ── */
            .mc-row { cursor: pointer; }
            .mc-row .rewards-full { display: none; }
            .mc-row.expanded .rewards-preview { display: none; }
            .mc-row.expanded .rewards-full { display: inline; word-break: break-word; }

            /* ── Event timeline ── */
            .event-list { display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }
            .event-item {
                display: flex;
                gap: 12px;
                align-items: flex-start;
                padding: 12px 14px;
                border-radius: 8px;
                border: 1px solid var(--border);
                background: var(--surface2);
            }
            .event-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
            .event-item h4 { font-size: 0.87rem; font-weight: 600; margin-bottom: 3px; }
            .event-item p { font-size: 0.82rem; color: var(--text2); }

            /* ── API links ── */
            .api-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
            .api-link {
                display: inline-block;
                padding: 6px 14px;
                background: var(--surface2);
                color: var(--accent);
                text-decoration: none;
                border: 1px solid var(--accent-border);
                border-radius: 999px;
                font-size: 0.83em;
                transition: background .15s;
            }
            .api-link:hover { background: var(--accent-light); }

            /* ── Quick start ── */
            pre {
                padding: 14px 16px;
                border-radius: 8px;
                overflow-x: auto;
                font-size: 0.8em;
                background: var(--surface2) !important;
                border: 1px solid var(--border);
                color: var(--text);
                font-family: 'Fira Mono', 'Consolas', monospace;
            }
            code { background: var(--surface2); padding: 1px 5px; border-radius: 4px; font-family: monospace; font-size: 0.88em; }

            /* ── Footer ── */
            .footer {
                text-align: center;
                color: var(--text2);
                font-size: 0.82rem;
                padding: 20px 0 0;
            }
            .footer a { color: var(--accent); text-decoration: none; }
        </style>
    </head>
    <body>
    <div class="page">

        <!-- Header -->
        <div class="header">
            <div>
                <h1>
                    Auditrix
                    <span class="feature-badge">⚡ Live</span>
                </h1>
                <p>OpenEnv-compliant AI agent benchmark for real-world compliance auditing — now with dynamic incident mechanics, structured explainability, and event-aware grading.</p>
            </div>
            <div class="header-right">
                <button class="theme-btn" id="themeBtn" title="Toggle dark mode" aria-label="Toggle dark mode"></button>
            </div>
        </div>

        <!-- Stat cards -->
        <div class="stats">
            <div class="stat">
                <div class="stat-n">7</div>
                <div class="stat-l">Tasks</div>
            </div>
            <div class="stat">
                <div class="stat-n">4</div>
                <div class="stat-l">Difficulty Tiers</div>
            </div>
            <div class="stat">
                <div class="stat-n">10</div>
                <div class="stat-l">Compliance Rules</div>
            </div>
            <div class="stat">
                <div class="stat-n">3</div>
                <div class="stat-l">Event Types</div>
            </div>
            <div class="stat">
                <div class="stat-n">62</div>
                <div class="stat-l">Tests Passing</div>
            </div>
            <div class="stat">
                <div class="stat-n">[0, 1]</div>
                <div class="stat-l">Score Range</div>
            </div>
        </div>

        <!-- Selected feature strip -->
        <div class="features">
            <span class="feat-label">Key Features:</span>
            <span class="feat new">Dynamic Events</span>
            <span class="feat new">evaluate_with_evidence()</span>
            <span class="feat new">Audit Confidence Report</span>
            <span class="feat new">Loop Detection</span>
            <span class="feat new">Variance Reporting (--seeds N)</span>
            <span class="feat new">Extreme Task</span>
            <span class="feat">R1-R10 Rules</span>
            <span class="feat">SOX · GDPR · FLSA</span>
            <span class="feat">OpenEnv Compatible</span>
        </div>

        <!-- Available Tasks -->
        <div class="section">
            <h2>Available Tasks</h2>
            <table>
                <thead>
                    <tr>
                        <th>Task ID</th>
                        <th>Difficulty</th>
                        <th>Records</th>
                        <th>Active Rules</th>
                        <th>Max Steps</th>
                        <th>Dynamic Events</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>easy_basic_audit</code></td>
                        <td><span class="pill pill-easy">🟢 Easy</span></td>
                        <td>5</td>
                        <td>R1, R2</td>
                        <td>25</td>
                        <td><span class="ev ev-outage">OUTAGE</span></td>
                    </tr>
                    <tr>
                        <td><code>medium_mixed_audit</code></td>
                        <td><span class="pill pill-medium">🟡 Medium</span></td>
                        <td>12</td>
                        <td>R1-R4</td>
                        <td>50</td>
                        <td><span class="ev ev-policy">POLICY</span> <span class="ev ev-outage">OUTAGE</span></td>
                    </tr>
                    <tr>
                        <td><code>hard_complex_audit</code></td>
                        <td><span class="pill pill-hard">🔴 Hard</span></td>
                        <td>20</td>
                        <td>R1-R5</td>
                        <td>100</td>
                        <td><span class="ev ev-amend">AMEND</span> <span class="ev ev-outage">OUTAGE</span></td>
                    </tr>
                    <tr>
                        <td><code>finance_sox_audit</code></td>
                        <td><span class="pill pill-hard">🔴 Hard</span></td>
                        <td>15</td>
                        <td>R3, R6-R8</td>
                        <td>80</td>
                        <td><span class="ev ev-policy">POLICY</span> <span class="ev ev-amend">AMEND</span> <span class="ev ev-outage">OUTAGE</span></td>
                    </tr>
                    <tr>
                        <td><code>gdpr_privacy_audit</code></td>
                        <td><span class="pill pill-medium">🟡 Medium</span></td>
                        <td>10</td>
                        <td>R5, R8, R9</td>
                        <td>50</td>
                        <td><span class="ev ev-amend">AMEND</span> <span class="ev ev-outage">OUTAGE</span></td>
                    </tr>
                    <tr>
                        <td><code>data_integrity_audit</code></td>
                        <td><span class="pill pill-medium">🟡 Medium</span></td>
                        <td>8</td>
                        <td>R3, R4, R10</td>
                        <td>40</td>
                        <td><span class="ev ev-amend">AMEND</span></td>
                    </tr>
                    <tr style="background: linear-gradient(90deg, var(--surface2), var(--surface));">
                        <td><code>regulatory_storm_audit</code> ⭐</td>
                        <td><span class="pill pill-extreme">🟣 Extreme</span></td>
                        <td>25</td>
                        <td>R1-R10 (all)</td>
                        <td>120</td>
                        <td><span class="ev ev-policy">POLICY</span> <span class="ev ev-policy">POLICY</span> <span class="ev ev-outage">OUTAGE</span> <span class="ev ev-outage">OUTAGE</span> <span class="ev ev-amend">AMEND</span> <span class="ev ev-amend">AMEND</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Dynamic Event Types -->
        <div class="section">
            <h2>⚡ Dynamic Event Types</h2>
            <div class="event-list">
                <div class="event-item">
                    <div class="event-icon">📜</div>
                    <div>
                        <h4><span class="ev ev-policy">POLICY_UPDATE</span> — Rule threshold changes mid-episode</h4>
                        <p>A rule's numeric threshold is updated (e.g. overtime approval threshold drops from 48 → 40 h/week). The agent must check <code>current_policy_overrides</code> each step and re-evaluate rules R1, R2, R7. Affects: medium, finance_sox, regulatory_storm tasks.</p>
                    </div>
                </div>
                <div class="event-item">
                    <div class="event-icon">🔒</div>
                    <div>
                        <h4><span class="ev ev-outage">SYSTEM_OUTAGE</span> — Record temporarily inaccessible</h4>
                        <p>A record is locked (legal hold, HR investigation, maintenance). Inspecting it returns an error with <code>outage_ends_at</code>. Agent must wait and retry after the window. Score not penalised for records in outage at episode end.</p>
                    </div>
                </div>
                <div class="event-item">
                    <div class="event-icon">✏️</div>
                    <div>
                        <h4><span class="ev ev-amend">RECORD_AMENDMENT</span> — Field value corrected mid-episode</h4>
                        <p>A field is corrected via a data ticket (e.g. <code>background_check: null → True</code>). If the amendment resolves a prior violation, flagging after the amendment is a <strong>false positive</strong> and costs −0.30. Agents must re-evaluate post-amendment.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Baseline Scores -->
        <div class="section">
            <h2>📊 Baseline Scores <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(Qwen 2.5 72B, temp=0, seed=42)</span></h2>
            <table>
                <thead>
                    <tr><th>Task</th><th>Difficulty</th><th>Score</th><th>Primary Failure Mode</th></tr>
                </thead>
                <tbody>
                    <tr><td><code>easy_basic_audit</code></td><td><span class="pill pill-easy">Easy</span></td><td><strong>0.92</strong></td><td><span class="fm-badge fm-none">none</span></td></tr>
                    <tr><td><code>medium_mixed_audit</code></td><td><span class="pill pill-medium">Medium</span></td><td><strong>0.75</strong></td><td><span class="fm-badge fm-false_positive">false_positive</span></td></tr>
                    <tr><td><code>hard_complex_audit</code></td><td><span class="pill pill-hard">Hard</span></td><td><strong>0.58</strong></td><td><span class="fm-badge fm-missed_violation">missed_violation</span></td></tr>
                    <tr><td><code>finance_sox_audit</code></td><td><span class="pill pill-hard">Hard</span></td><td><strong>0.61</strong></td><td><span class="fm-badge fm-report_inconsistency">report_inconsistency</span></td></tr>
                    <tr><td><code>gdpr_privacy_audit</code></td><td><span class="pill pill-medium">Medium</span></td><td><strong>0.72</strong></td><td><span class="fm-badge fm-false_positive">false_positive</span></td></tr>
                    <tr><td><code>data_integrity_audit</code></td><td><span class="pill pill-medium">Medium</span></td><td><strong>0.74</strong></td><td><span class="fm-badge fm-missed_violation">missed_violation</span></td></tr>
                    <tr><td><code>regulatory_storm_audit</code> ⭐</td><td><span class="pill pill-extreme">Extreme</span></td><td><strong>0.31</strong></td><td><span class="fm-badge fm-low_coverage">low_coverage</span></td></tr>
                    <tr class="avg-row"><td>Overall Average</td><td></td><td><strong>0.66</strong></td><td></td></tr>
                </tbody>
            </table>
        </div>

        <!-- Model Comparison -->
        <div class="section">
            <h2>Model Comparison <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(click row to expand rewards)</span></h2>
            <table id="mcTable" class="mc-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Task</th>
                        <th>Model</th>
                        <th>Seed</th>
                        <th>Score</th>
                        <th>Steps</th>
                        <th>Success</th>
                        <th>Failure Mode</th>
                        <th>Rewards</th>
                    </tr>
                </thead>
                <tbody>
                    __MODEL_COMPARISON_ROWS__
                </tbody>
            </table>
        </div>

        <!-- API Endpoints -->
        <div class="section">
            <h2>🔌 API Endpoints</h2>
            <div class="api-links">
                <a href="/health" class="api-link">GET /health</a>
                <a href="/tasks" class="api-link">GET /tasks</a>
                <a href="/baseline" class="api-link">GET /baseline</a>
                <a href="/state" class="api-link">GET /state</a>
                <a href="/docs" class="api-link">Interactive Docs</a>
            </div>
            <p style="margin-top:12px;font-size:0.83rem;color:var(--text2)">POST: <code>/reset</code> &nbsp;·&nbsp; <code>/step</code></p>
        </div>

        <!-- Quick Start -->
        <div class="section">
            <h2>Quick Start</h2>
            <p style="margin-bottom:10px;color:var(--text2);font-size:0.85rem;">Run an episode with dynamic event awareness:</p>
            <pre># 1. Reset with a seed (for reproducible event schedule)
curl -X POST http://localhost:7860/reset \\
  -H "Content-Type: application/json" \\
  -d '{"task_id": "regulatory_storm_audit", "seed": 42}'

# 2. Inspect a record
curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{"action_type": "inspect_record", "record_id": "RS001"}'

# 3. Apply a rule (returns reason_codes + evidence)
curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{"action_type": "apply_rule", "record_id": "RS001", "rule_id": "R1"}'

# 4. Submit final report with audit_confidence section
curl -X POST http://localhost:7860/step \\
  -H "Content-Type: application/json" \\
  -d '{
    "action_type": "generate_report",
    "report": {
      "summary": "Found 12 violations across 25 records.",
      "flagged_violations": [{"record_id": "RS001", "rule_id": "R1"}],
      "compliant_records": ["RS021", "RS025"],
      "recommendations": ["Re-check records after POLICY_UPDATE events."],
      "audit_confidence": {
        "evidence_coverage_ratio": 0.92,
        "high_confidence_flags": ["RS001:R1", "RS002:R2"],
        "uncertain_flags": ["RS007:R6"],
        "reasoning": "Cross-checked all fields against active policy overrides."
      }
    }
  }'</pre>
        </div>

        <div class="footer">
            <p>Auditrix &nbsp;·&nbsp; <a href="https://github.com/Kowshikv07/Auditrix">GitHub</a> &nbsp;·&nbsp; OpenEnv Framework &nbsp;·&nbsp; 62 tests passing  </p>
        </div>

    </div><!-- /page -->

    <script>
        // Dark mode
        const btn = document.getElementById('themeBtn');
        if (localStorage.getItem('theme') === 'dark') document.body.classList.add('dark');
        btn.addEventListener('click', () => {
            document.body.classList.toggle('dark');
            localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light');
        });

        // Expandable model comparison rows
        const rows = document.querySelectorAll('#mcTable .mc-row');
        const collapseAll = () => rows.forEach(r => r.classList.remove('expanded'));
        rows.forEach(r => r.addEventListener('click', e => {
            e.stopPropagation();
            const was = r.classList.contains('expanded');
            collapseAll();
            if (!was) r.classList.add('expanded');
        }));
        document.addEventListener('click', e => {
            if (!e.target.closest('#mcTable')) collapseAll();
        });
    </script>
    </body>
    </html>
    """
    html_content = html_content.replace("__MODEL_COMPARISON_ROWS__", model_rows)
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
