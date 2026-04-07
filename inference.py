from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List

from openai import OpenAI

from openenv_ticket_triage.environment import TicketTriageEnv
from openenv_ticket_triage.models import TicketTriageAction
from openenv_ticket_triage.tasks import TASKS


SEED = 7
MAX_STEPS_DEFAULT = 24

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")


SYSTEM_PROMPT = (
    "You are a customer-support operations agent. "
    "Produce one JSON action at a time to triage tickets. "
    "Allowed action_type values: inspect_ticket, set_priority, assign_team, "
    "request_customer_reply, add_internal_note, escalate_compliance, resolve_ticket, finish. "
    "Always output valid JSON with keys: action_type, ticket_id, value."
)


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not brace:
            raise
        return json.loads(brace.group(0))


def _fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    tickets = observation.get("visible_tickets", [])
    for t in tickets:
        if not t.get("inspected"):
            return {"action_type": "inspect_ticket", "ticket_id": t["ticket_id"], "value": None}
    for t in tickets:
        if not t.get("priority"):
            return {"action_type": "set_priority", "ticket_id": t["ticket_id"], "value": "medium"}
    for t in tickets:
        if not t.get("assigned_team"):
            return {"action_type": "assign_team", "ticket_id": t["ticket_id"], "value": "support"}
    for t in tickets:
        if not t.get("resolved"):
            return {"action_type": "resolve_ticket", "ticket_id": t["ticket_id"], "value": "resolved"}
    return {"action_type": "finish", "ticket_id": None, "value": None}


def choose_action(client: OpenAI, observation: Dict[str, Any], seed: int) -> Dict[str, Any]:
    user_prompt = {
        "objective": observation.get("objective"),
        "step_index": observation.get("step_index"),
        "max_steps": observation.get("max_steps"),
        "progress_score": observation.get("progress_score"),
        "tickets": observation.get("visible_tickets", []),
        "recent_actions": observation.get("action_history", [])[-6:],
        "last_action_error": observation.get("last_action_error"),
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        temperature=0,
        max_tokens=160,
        seed=seed,
    )
    content = response.choices[0].message.content or ""
    action_payload = _extract_json(content)

    # Ensure required keys exist.
    action_payload.setdefault("ticket_id", None)
    action_payload.setdefault("value", None)
    return action_payload


def run_task(client: OpenAI, task_id: str) -> float:
    env = TicketTriageEnv(task_id=task_id)
    obs = env.reset().model_dump()
    done = False
    step_idx = 0
    rewards: List[float] = []
    success = False
    score = 0.0

    print(f"[START] task={task_id} env={env.benchmark_name} model={MODEL_NAME}")

    try:
        while not done and step_idx < MAX_STEPS_DEFAULT:
            step_idx += 1
            error_msg = None

            try:
                payload = choose_action(client, obs, seed=SEED + step_idx)
            except Exception:
                payload = _fallback_action(obs)

            try:
                action = TicketTriageAction.model_validate(payload)
            except Exception:
                payload = _fallback_action(obs)
                action = TicketTriageAction.model_validate(payload)

            result = env.step(action)
            obs = result.observation.model_dump()
            done = result.done
            reward = float(result.reward)
            rewards.append(reward)

            error_msg = obs.get("last_action_error")
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            print(
                f"[STEP] step={step_idx} action={action_str} reward={_fmt_reward(reward)} "
                f"done={_bool(done)} error={error_msg if error_msg else 'null'}"
            )

            score = float(result.info.get("task_score", 0.0))

        final_state = env.state()
        success = score >= 0.7 and final_state.done
    except Exception:
        success = False
    finally:
        rewards_str = ",".join(_fmt_reward(v) for v in rewards)
        print(
            f"[END] success={_bool(success)} steps={step_idx} score={score:.2f} rewards={rewards_str}"
        )

    return score


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY (preferred), or HF_TOKEN/API_KEY for router usage."
        )

    random.seed(SEED)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id)
    _ = sum(scores.values()) / len(scores)


if __name__ == "__main__":
    main()
