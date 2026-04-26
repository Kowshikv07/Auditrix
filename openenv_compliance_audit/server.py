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
    baseline_rows = ""
    task_rows = ""
    leaderboard_rows = ""
    parsed_runs: List[Dict[str, Any]] = []
    latest_by_task: Dict[str, Dict[str, Any]] = {}
    latest_timestamp = "-"
    models: set[str] = set()

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

                parsed_runs.append(item)
                ts = str(item.get("timestamp", ""))
                if ts and (latest_timestamp == "-" or ts > latest_timestamp):
                    latest_timestamp = ts

                task_id = str(item.get("task_id", ""))
                model_name = str(item.get("model", ""))
                if model_name:
                    models.add(model_name)
                if task_id:
                    prev = latest_by_task.get(task_id)
                    prev_ts = str(prev.get("timestamp", "")) if prev else ""
                    if prev is None or ts > prev_ts:
                        latest_by_task[task_id] = item

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
                failure_slug = str(failure_mode).lower().replace(" ", "_")

                model_rows += (
                    "<tr class=\"mc-row\" "
                    f"data-task=\"{escape(task_id.lower())}\" "
                    f"data-model=\"{escape(model_name.lower())}\" "
                    f"data-failure=\"{escape(failure_slug)}\">"
                    f"<td>{escape(str(item.get('timestamp', '')))}</td>"
                    f"<td>{escape(str(item.get('task_id', '')))}</td>"
                    f"<td>{escape(str(item.get('model', '')))}</td>"
                    f"<td>{escape(str(seed_val))}</td>"
                    f"<td>{escape(str(item.get('score', '')))}</td>"
                    f"<td>{escape(str(item.get('steps', '')))}</td>"
                    f"<td>{escape(str(item.get('success', '')))}</td>"
                    f"<td><span class=\"fm-badge fm-{escape(failure_slug)}\">{escape(str(failure_mode))}</span></td>"
                    "<td class=\"rewards-cell\">"
                    f"<span class=\"rewards-preview\">{escape(rewards_preview)}</span>"
                    f"<span class=\"rewards-full\">{escape(rewards_full)}</span>"
                    "</td>"
                    "</tr>"
                )

    if not model_rows:
        model_rows = "<tr><td colspan=\"9\">No benchmark records found. Run <code>python inference.py</code> to generate data.</td></tr>"

    task_count = len(TASKS)
    difficulty_count = len({task.difficulty for task in TASKS.values()})
    rule_count = len({rid for task in TASKS.values() for rid in task.active_rule_ids})
    total_runs = len(parsed_runs)
    mean_score = (
        sum(float(item.get("score", 0.0)) for item in parsed_runs) / total_runs
        if total_runs
        else 0.0
    )
    success_rate = (
        (sum(1 for item in parsed_runs if item.get("success") is True) / total_runs) * 100.0
        if total_runs
        else 0.0
    )
    top_score = max((float(item.get("score", 0.0)) for item in parsed_runs), default=0.0)

    per_model: Dict[str, Dict[str, float]] = {}
    for item in parsed_runs:
        m = str(item.get("model", "unknown"))
        score = float(item.get("score", 0.0))
        bucket = per_model.setdefault(m, {"count": 0.0, "total": 0.0, "best": 0.0})
        bucket["count"] += 1.0
        bucket["total"] += score
        bucket["best"] = max(bucket["best"], score)

    ranked_models = sorted(
        per_model.items(),
        key=lambda kv: (kv[1]["total"] / kv[1]["count"]) if kv[1]["count"] else 0.0,
        reverse=True,
    )
    for idx, (model_name, stats) in enumerate(ranked_models[:8], start=1):
        avg = (stats["total"] / stats["count"]) if stats["count"] else 0.0
        leaderboard_rows += (
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{escape(model_name)}</td>"
            f"<td><strong>{avg:.3f}</strong></td>"
            f"<td>{int(stats['count'])}</td>"
            f"<td>{stats['best']:.3f}</td>"
            "</tr>"
        )
    if not leaderboard_rows:
        leaderboard_rows = "<tr><td colspan=\"5\">No model leaderboard data yet.</td></tr>"

    difficulty_labels = {
        "easy": ("pill-easy", "Easy"),
        "medium": ("pill-medium", "Medium"),
        "hard": ("pill-hard", "Hard"),
        "extreme": ("pill-extreme", "Extreme"),
        "streaming": ("pill-extreme", "Streaming"),
    }

    def _event_badges(task_id: str) -> str:
        event_types: set[str] = set()
        try:
            probe = ComplianceAuditEnv(task_id=task_id)
            probe.reset(seed=42)
            for ev in probe.state().event_schedule:
                event_types.add(str(ev.event_type.value))
        except Exception:
            pass

        if not event_types:
            return "<span class=\"ev\">NONE</span>"

        badges = []
        for ev_type in sorted(event_types):
            if "policy" in ev_type:
                badges.append('<span class="ev ev-policy">POLICY</span>')
            elif "outage" in ev_type:
                badges.append('<span class="ev ev-outage">OUTAGE</span>')
            elif "amend" in ev_type:
                badges.append('<span class="ev ev-amend">AMEND</span>')
            elif "susp" in ev_type:
                badges.append('<span class="ev ev-policy">SUSPEND</span>')
            else:
                badges.append(f'<span class="ev">{escape(ev_type.upper())}</span>')
        return " ".join(badges)

    for t in TASKS.values():
        pill_class, pill_label = difficulty_labels.get(t.difficulty, ("pill-medium", t.difficulty.title()))
        rules_text = ", ".join(t.active_rule_ids)
        task_rows += (
            "<tr>"
            f"<td><code>{escape(t.task_id)}</code></td>"
            f"<td><span class=\"pill {pill_class}\">{escape(pill_label)}</span></td>"
            f"<td>{len(t.records)}</td>"
            f"<td>{escape(rules_text)}</td>"
            f"<td>{t.max_steps}</td>"
            f"<td>{_event_badges(t.task_id)}</td>"
            "</tr>"
        )

    if not task_rows:
        task_rows = "<tr><td colspan=\"6\">No task definitions found.</td></tr>"

    for task_id, task in TASKS.items():
        pill_class, pill_label = difficulty_labels.get(task.difficulty, ("pill-medium", task.difficulty.title()))
        latest = latest_by_task.get(task_id)
        if latest is None:
            baseline_rows += (
                "<tr>"
                f"<td><code>{escape(task_id)}</code></td>"
                f"<td><span class=\"pill {pill_class}\">{escape(pill_label)}</span></td>"
                "<td><strong>-</strong></td>"
                "<td>-</td>"
                "</tr>"
            )
            continue

        score = float(latest.get("score", 0.0))
        failure_mode = str(latest.get("failure_mode", "none"))
        failure_slug = failure_mode.lower().replace(" ", "_")
        baseline_rows += (
            "<tr>"
            f"<td><code>{escape(task_id)}</code></td>"
            f"<td><span class=\"pill {pill_class}\">{escape(pill_label)}</span></td>"
            f"<td><strong>{score:.3f}</strong></td>"
            f"<td><span class=\"fm-badge fm-{escape(failure_slug)}\">{escape(failure_mode)}</span></td>"
            "</tr>"
        )

    if latest_by_task:
        latest_scores = [float(item.get("score", 0.0)) for item in latest_by_task.values()]
        baseline_rows += (
            "<tr class=\"avg-row\">"
            "<td>Overall Average</td>"
            "<td></td>"
            f"<td><strong>{(sum(latest_scores) / len(latest_scores)):.3f}</strong></td>"
            "<td></td>"
            "</tr>"
        )

    task_options = "".join(
        f'<option value="{escape(task_id.lower())}">{escape(task_id)}</option>'
        for task_id in TASKS.keys()
    )
    model_options = "".join(
        f'<option value="{escape(model_name.lower())}">{escape(model_name)}</option>'
        for model_name in sorted(models)
    )

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
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
            :root {
                --bg: #f0f2f5;
                --surface: #ffffff;
                --surface2: #f8f9fc;
                --border: #e2e5ea;
                --text: #1a1d23;
                --text2: #5c6370;
                --accent: #0f766e;
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
                --accent: #2dd4bf;
                --accent-light: #1a2240;
                --accent-border: #2e4180;
                --green: #4ade80;
                --yellow: #fbbf24;
                --red: #f87171;
                --purple: #a78bfa;
                --orange: #fb923c;
            }
            body {
                font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg);
                color: var(--text);
                min-height: 100vh;
                padding: 24px 16px 40px;
                transition: background .3s, color .3s;
                font-size: 14px;
            }
            .page { max-width: 1360px; margin: 0 auto; }

            .tabs {
                display: inline-flex;
                gap: 6px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 4px;
                margin-bottom: 18px;
                box-shadow: var(--shadow);
            }
            .tab-btn {
                border: 0;
                background: transparent;
                color: var(--text2);
                font-size: 0.82rem;
                font-weight: 700;
                padding: 8px 16px;
                border-radius: 999px;
                cursor: pointer;
                transition: all .2s;
            }
            .tab-btn.active {
                background: var(--accent-light);
                color: var(--accent);
                border: 1px solid var(--accent-border);
            }
            .tab-panel { display: none; }
            .tab-panel.active { display: block; }

            .overview-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 14px;
                margin-bottom: 20px;
            }
            .overview-card {
                background: linear-gradient(160deg, var(--surface), var(--surface2));
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 16px 18px;
                box-shadow: var(--shadow);
            }
            .overview-card .k { font-size: 0.74rem; color: var(--text2); text-transform: uppercase; letter-spacing: .5px; }
            .overview-card .v { font-size: 1.5rem; margin-top: 4px; font-weight: 800; color: var(--accent); }
            .overview-card .s { font-size: 0.8rem; margin-top: 5px; color: var(--text2); }

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
            .split {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
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
            .mc-row.hidden { display: none; }
            .mc-row .rewards-full { display: none; }
            .mc-row.expanded .rewards-preview { display: none; }
            .mc-row.expanded .rewards-full { display: inline; word-break: break-word; }

            .controls {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 12px;
                align-items: center;
            }
            .controls select {
                border: 1px solid var(--border);
                background: var(--surface2);
                color: var(--text);
                border-radius: 8px;
                padding: 7px 10px;
                font-size: 0.82rem;
            }
            .controls .meta {
                color: var(--text2);
                font-size: 0.8rem;
                margin-left: auto;
            }

            .score-chip {
                display: inline-block;
                border-radius: 999px;
                padding: 3px 10px;
                font-size: 0.74rem;
                font-weight: 700;
                background: var(--accent-light);
                color: var(--accent);
                border: 1px solid var(--accent-border);
            }

            .sim-grid {
                display: grid;
                grid-template-columns: 320px 1fr;
                gap: 16px;
            }
            .sim-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 16px;
                box-shadow: var(--shadow);
            }
            .sim-card h3 {
                font-size: 0.92rem;
                margin-bottom: 10px;
            }
            .sim-field {
                display: flex;
                flex-direction: column;
                gap: 5px;
                margin-bottom: 10px;
            }
            .sim-field label {
                font-size: 0.76rem;
                color: var(--text2);
                text-transform: uppercase;
                letter-spacing: .5px;
                font-weight: 700;
            }
            .sim-field input,
            .sim-field select,
            .sim-field textarea {
                border: 1px solid var(--border);
                background: var(--surface2);
                color: var(--text);
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 0.82rem;
                width: 100%;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
            }
            .sim-field textarea {
                min-height: 130px;
                resize: vertical;
            }
            .sim-actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                margin-top: 8px;
            }
            .sim-btn {
                border: 1px solid var(--accent-border);
                background: var(--accent-light);
                color: var(--accent);
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 0.8rem;
                font-weight: 700;
                cursor: pointer;
            }
            .sim-btn.secondary {
                border-color: var(--border);
                background: var(--surface2);
                color: var(--text);
            }
            .sim-output {
                min-height: 460px;
                white-space: pre-wrap;
                overflow: auto;
            }

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
                font-family: 'JetBrains Mono', 'Consolas', monospace;
            }
            code { background: var(--surface2); padding: 1px 5px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.88em; }

            @media (max-width: 1024px) {
                .split { grid-template-columns: 1fr; }
                .controls .meta { width: 100%; margin-left: 0; }
                .sim-grid { grid-template-columns: 1fr; }
            }

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
                <p>OpenEnv-compliant AI agent benchmark for compliance auditing with dynamic incidents, evidence-aware grading, and long-horizon RL evaluation.</p>
            </div>
            <div class="header-right">
                <button class="theme-btn" id="themeBtn" title="Toggle dark mode" aria-label="Toggle dark mode"></button>
            </div>
        </div>

        <div class="tabs" role="tablist" aria-label="Dashboard tabs">
            <button class="tab-btn active" data-tab="report" id="tabReport">Report</button>
            <button class="tab-btn" data-tab="simulation" id="tabSimulation">Simulation</button>
        </div>

        <div class="tab-panel active" id="panel-report">
        <div class="overview-grid">
            <div class="overview-card">
                <div class="k">Latest Log Timestamp</div>
                <div class="v">__LATEST_TIMESTAMP__</div>
                <div class="s">Auto-read from inference_runs.jsonl</div>
            </div>
            <div class="overview-card">
                <div class="k">Best Recorded Score</div>
                <div class="v">__TOP_SCORE__</div>
                <div class="s">Highest score across all recorded runs</div>
            </div>
            <div class="overview-card">
                <div class="k">Unique Models</div>
                <div class="v">__MODEL_COUNT__</div>
                <div class="s">Models compared in this dashboard</div>
            </div>
        </div>

        <!-- Stat cards -->
        <div class="stats">
            <div class="stat">
                <div class="stat-n">__TASK_COUNT__</div>
                <div class="stat-l">Tasks</div>
            </div>
            <div class="stat">
                <div class="stat-n">__DIFFICULTY_COUNT__</div>
                <div class="stat-l">Difficulty Tiers</div>
            </div>
            <div class="stat">
                <div class="stat-n">__RULE_COUNT__</div>
                <div class="stat-l">Compliance Rules</div>
            </div>
            <div class="stat">
                <div class="stat-n">__RUN_COUNT__</div>
                <div class="stat-l">Recorded Runs</div>
            </div>
            <div class="stat">
                <div class="stat-n">__MEAN_SCORE__</div>
                <div class="stat-l">Mean Score</div>
            </div>
            <div class="stat">
                <div class="stat-n">__SUCCESS_RATE__</div>
                <div class="stat-l">Success Rate</div>
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
            <span class="feat">R1-R11 Rules</span>
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
                    __TASK_ROWS__
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

        <div class="split">
            <div class="section">
                <h2>📊 Latest Scores By Task <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(most recent run per task)</span></h2>
                <table>
                    <thead>
                        <tr><th>Task</th><th>Difficulty</th><th>Score</th><th>Primary Failure Mode</th></tr>
                    </thead>
                    <tbody>
                        __BASELINE_ROWS__
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h2>🏁 Model Leaderboard <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(top by average score)</span></h2>
                <table>
                    <thead>
                        <tr><th>#</th><th>Model</th><th>Avg Score</th><th>Runs</th><th>Best</th></tr>
                    </thead>
                    <tbody>
                        __LEADERBOARD_ROWS__
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Model Comparison -->
        <div class="section">
            <h2>Model Comparison <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(click row to expand rewards)</span></h2>
            <div class="controls">
                <select id="taskFilter">
                    <option value="">All tasks</option>
                    __TASK_OPTIONS__
                </select>
                <select id="modelFilter">
                    <option value="">All models</option>
                    __MODEL_OPTIONS__
                </select>
                <div class="meta"><span class="score-chip">Mean __MEAN_SCORE__ · Success __SUCCESS_RATE__</span></div>
            </div>
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
        </div>

        <div class="tab-panel" id="panel-simulation">
            <div class="section">
                <h2>🧪 Simulation Playground <span style="font-weight:400;font-size:0.83rem;color:var(--text2)">(interactive reset / step)</span></h2>
                <div class="sim-grid">
                    <div class="sim-card">
                        <h3>Episode Controls</h3>
                        <div class="sim-field">
                            <label for="simTask">Task</label>
                            <select id="simTask">
                                __TASK_OPTIONS__
                            </select>
                        </div>
                        <div class="sim-field">
                            <label for="simSeed">Seed</label>
                            <input id="simSeed" type="number" value="42" />
                        </div>
                        <div class="sim-actions">
                            <button class="sim-btn" id="simResetBtn">Reset Episode</button>
                            <button class="sim-btn secondary" id="simHealthBtn">Check Health</button>
                        </div>

                        <h3 style="margin-top:14px;">Step Action</h3>
                        <div class="sim-field">
                            <label for="simAction">Action JSON</label>
                            <textarea id="simAction">{"action_type":"inspect_record","record_id":"E001"}</textarea>
                        </div>
                        <div class="sim-actions">
                            <button class="sim-btn" id="simStepBtn">Send Step</button>
                            <button class="sim-btn secondary" id="simClearBtn">Clear Output</button>
                        </div>
                    </div>

                    <div class="sim-card">
                        <h3>Simulation Output</h3>
                        <pre class="sim-output" id="simOutput">Ready. Use "Reset Episode" to start.</pre>
                    </div>
                </div>
            </div>
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

        // Filters
        const taskFilter = document.getElementById('taskFilter');
        const modelFilter = document.getElementById('modelFilter');

        const applyFilters = () => {
            const taskVal = (taskFilter?.value || '').trim();
            const modelVal = (modelFilter?.value || '').trim();

            rows.forEach(row => {
                const taskMatch = !taskVal || row.dataset.task === taskVal;
                const modelMatch = !modelVal || row.dataset.model === modelVal;
                row.classList.toggle('hidden', !(taskMatch && modelMatch));
            });
        };

        taskFilter?.addEventListener('change', applyFilters);
        modelFilter?.addEventListener('change', applyFilters);
        applyFilters();

        // Tab switching
        const tabButtons = document.querySelectorAll('.tab-btn');
        const panels = {
            report: document.getElementById('panel-report'),
            simulation: document.getElementById('panel-simulation')
        };
        tabButtons.forEach(btnEl => {
            btnEl.addEventListener('click', () => {
                const tab = btnEl.dataset.tab;
                tabButtons.forEach(b => b.classList.remove('active'));
                btnEl.classList.add('active');
                Object.values(panels).forEach(p => p?.classList.remove('active'));
                panels[tab]?.classList.add('active');
            });
        });

        // Simulation API playground
        const simTask = document.getElementById('simTask');
        const simSeed = document.getElementById('simSeed');
        const simAction = document.getElementById('simAction');
        const simOutput = document.getElementById('simOutput');
        const simResetBtn = document.getElementById('simResetBtn');
        const simStepBtn = document.getElementById('simStepBtn');
        const simHealthBtn = document.getElementById('simHealthBtn');
        const simClearBtn = document.getElementById('simClearBtn');

        const writeSim = (title, payload) => {
            const ts = new Date().toISOString();
            const text = `[${ts}] ${title}\n${typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2)}\n\n`;
            simOutput.textContent = text + simOutput.textContent;
        };

        const callJson = async (url, options = {}) => {
            const res = await fetch(url, options);
            const text = await res.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch {
                data = text;
            }
            return { ok: res.ok, status: res.status, data };
        };

        simResetBtn?.addEventListener('click', async () => {
            const body = {
                task_id: (simTask?.value || '').trim() || null,
                seed: Number(simSeed?.value || 42)
            };
            const res = await callJson('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            writeSim(`POST /reset -> ${res.status}`, res.data);
        });

        simStepBtn?.addEventListener('click', async () => {
            let actionBody;
            try {
                actionBody = JSON.parse(simAction?.value || '{}');
            } catch (e) {
                writeSim('Invalid action JSON', String(e));
                return;
            }
            const res = await callJson('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(actionBody)
            });
            writeSim(`POST /step -> ${res.status}`, res.data);
        });

        simHealthBtn?.addEventListener('click', async () => {
            const res = await callJson('/health');
            writeSim(`GET /health -> ${res.status}`, res.data);
        });

        simClearBtn?.addEventListener('click', () => {
            simOutput.textContent = 'Cleared.';
        });
    </script>
    </body>
    </html>
    """
    html_content = html_content.replace("__TASK_COUNT__", str(task_count))
    html_content = html_content.replace("__DIFFICULTY_COUNT__", str(difficulty_count))
    html_content = html_content.replace("__RULE_COUNT__", str(rule_count))
    html_content = html_content.replace("__RUN_COUNT__", str(total_runs))
    html_content = html_content.replace("__MEAN_SCORE__", f"{mean_score:.3f}")
    html_content = html_content.replace("__SUCCESS_RATE__", f"{success_rate:.1f}%")
    html_content = html_content.replace("__TASK_ROWS__", task_rows)
    html_content = html_content.replace("__BASELINE_ROWS__", baseline_rows)
    html_content = html_content.replace("__TASK_OPTIONS__", task_options)
    html_content = html_content.replace("__MODEL_OPTIONS__", model_options)
    html_content = html_content.replace("__TOP_SCORE__", f"{top_score:.3f}")
    html_content = html_content.replace("__MODEL_COUNT__", str(len(models)))
    html_content = html_content.replace("__LEADERBOARD_ROWS__", leaderboard_rows)
    html_content = html_content.replace("__LATEST_TIMESTAMP__", escape(latest_timestamp))
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
