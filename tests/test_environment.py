from openenv_ticket_triage.environment import TicketTriageEnv
from openenv_ticket_triage.models import TicketTriageAction


def test_reset_and_state_are_consistent() -> None:
    env = TicketTriageEnv(task_id="easy_refund_priority")
    obs = env.reset()
    state = env.state()

    assert obs.task_id == "easy_refund_priority"
    assert state.task_id == "easy_refund_priority"
    assert state.step_count == 0
    assert state.done is False


def test_deterministic_easy_task_full_score() -> None:
    env = TicketTriageEnv(task_id="easy_refund_priority")
    env.reset()

    env.step(TicketTriageAction(action_type="inspect_ticket", ticket_id="T-1001"))
    env.step(TicketTriageAction(action_type="set_priority", ticket_id="T-1001", value="high"))
    env.step(TicketTriageAction(action_type="assign_team", ticket_id="T-1001", value="billing"))
    env.step(TicketTriageAction(action_type="resolve_ticket", ticket_id="T-1001", value="resolved"))
    result = env.step(TicketTriageAction(action_type="finish"))

    assert result.done is True
    assert 0.95 <= float(result.info["task_score"]) <= 1.0


def test_invalid_action_is_penalized() -> None:
    env = TicketTriageEnv(task_id="easy_refund_priority")
    env.reset()

    result = env.step(TicketTriageAction(action_type="set_priority", ticket_id="T-1001", value="unknown"))
    assert result.reward == 0.0
    assert result.observation.last_action_error is not None
