# Auditrix: Training LLMs to Actually Do Compliance Audits

![Auditrix Logo](assets/Auditrix-logo.png)

## Architecture Diagram

![Auditrix Architecture](assets/Auditrix.png)

## Why We Built This

This project started from one uncomfortable observation.

When we asked strong LLMs to do compliance audits, the answers looked polished and confident, but the decision quality was often brittle. They would miss cross-record violations, over-flag exempt cases, or produce final reports that contradicted their own earlier actions.

That gap mattered to us because compliance is not a style task. It is a decision task.

In real audits, every false positive burns trust, every missed violation can be expensive, and every final report must be traceable. So instead of asking an LLM to "write an audit," we built an environment where the model has to behave like an auditor under constraints.

That environment is Auditrix.

Our north star was simple: if an agent can learn to make reliable, auditable decisions in this setting, we are moving from fluent text generation toward operational reasoning.

---

## One-Line Summary

Auditrix is an OpenEnv-compatible reinforcement learning environment for enterprise compliance auditing, where agents must inspect records, apply rules, gather evidence, flag or retract violations, and submit a consistent final report under a strict step budget.

---

## Hackathon Positioning

### Primary Theme: #3.1 World Modeling (Professional Tasks)

Auditrix is built around a realistic professional workflow where the world evolves during execution.
The agent must act under partial observability, use deterministic tools, and maintain a coherent state over many decisions.
Performance is evaluated on accountable actions, not polished one-shot text outputs.

### Secondary Theme: #2 Long-Horizon Planning & Instruction Following

Episodes are long, constrained by strict step budgets, and include delayed consequences.
The streaming task emphasizes sparse terminal-heavy reward, while dynamic events force online re-planning.
The required `prioritize_rules` action also tests whether declared strategy is actually followed in behavior.

### What This Demonstrates

- Real professional-task competence instead of single-turn answer quality
- Strategic planning and triage under hard budget constraints
- Robust adaptation to policy changes, outages, and record amendments
- Reproducible, diagnosable evaluation through structured metrics and failure modes

---

## What the Agent Can Actually Do

At each step, the agent selects one action from a constrained interface:

- `inspect_record`
- `apply_rule`
- `request_evidence`
- `flag_violation`
- `retract_flag`
- `mark_compliant`
- `prioritize_rules`
- `generate_report`

There are no hidden shortcuts and no free-form "explain your reasoning" loopholes for reward hacking. Every action is explicit, logged, and gradable.

---

## Task Inventory

Auditrix currently includes 8 tasks, intentionally ordered from clean starter scenarios to high-noise operational stress tests:

- `easy_basic_audit`
- `medium_mixed_audit`
- `hard_complex_audit`
- `finance_sox_audit`
- `gdpr_privacy_audit`
- `data_integrity_audit`
- `regulatory_storm_audit`
- `streaming_long_horizon`

### Difficulty progression

- Easy and medium tasks teach the agent to ground actions in rules and evidence.
- Hard and domain audits introduce ambiguity, overlapping violations, and practical tradeoffs.
- Extreme and streaming tasks force long-horizon planning under changing state and tighter budget pressure.

### Long-horizon stress case

- `streaming_long_horizon`
- 350 records
- 400-step budget
- all major rule families active
- designed for strategic sampling, memory, and delayed-reward robustness

This task is where weak policies usually collapse: either they over-inspect and run out of budget, or they under-inspect and miss critical violations.

---

## Rule Coverage and Why It Is Non-Trivial

Auditrix uses a dynamic rule framework.
Each task activates a different subset of rule families, and runtime policy updates can change how rules are evaluated mid-episode.
The current benchmark ships with a base catalog of rules, but the environment is designed to scale to additional rules without changing the agent interface.

Rule families include:

- minor and intern overtime constraints
- salary band validation by role
- duplicate employee identifiers
- contract and eligibility checks
- compliance training requirements
- GDPR consent and field integrity
- manager/reference consistency checks

The complexity comes from interaction effects. A rule that appears clear in isolation can flip under exemptions, policy updates, or cross-record evidence. That means agents must maintain live context, not just run static if-else checks.

---

## Why This Environment Is Genuinely Hard

### 1) Exemption-sensitive precision

The correct action is often "do not flag." Agents that over-generalize patterns collapse on precision.

### 2) Cross-record state tracking

Some violations only emerge from comparisons across records. Single-record heuristics fail.

### 3) Dynamic world updates

Mid-episode events can alter policy thresholds, amend records, or suspend access. Strategies must adapt online.

### 4) Long-horizon budget pressure

Inspect-everything behavior does not scale. Agents must prioritize, triage, and recover from early mistakes.

### 5) Report accountability

Final reports are judged for consistency with trajectory state, not just presence or formatting.

In practice, this is the key difference from many prompt-only benchmarks: the model is evaluated on the full chain of decisions, not on how persuasive the final paragraph sounds.

---

## Technical Deep Dive

### Environment mechanics

OpenEnv-style lifecycle:

- `reset(task_id, seed)` initializes records and deterministic event schedule
- `step(action)` applies action effects, reward shaping, event updates, and termination logic
- `state()` exposes auditable internal episode state for evaluation and debugging

Dynamic events include:

- `POLICY_UPDATE`
- `SYSTEM_OUTAGE`
- `RECORD_AMENDMENT`

### A typical episode (what actually happens)

1. The agent declares strategy via `prioritize_rules`.
2. It inspects a focused subset of records and runs targeted rule checks.
3. Mid-episode, a policy threshold may change or records may be amended.
4. The agent must revise earlier assumptions, retract or add flags, and preserve precision.
5. It generates a final report that is checked against the true trajectory state.

If the report disagrees with what the agent actually did, the system penalizes that inconsistency.

### Reward model

Per-step reward is clamped to [-1, 1].

Core design combines action-level shaping with terminal quality signals:

- positive shaping for useful progress
- penalties for redundant or exploit-like loops
- terminal bonus tied to task score
- optional report-quality bonus
- consistency penalty if submitted report contradicts environment state

This lets us reward good local behavior without losing focus on final audit quality.

### Grader model

Terminal task score is normalized to [0, 1] and computed from weighted components:

- severity-weighted detection
- false-positive behavior
- coverage
- efficiency
- prioritization quality
- warning and abstention behavior
- anti-loop deductions

Difficulty-specific graders adjust weights and caps to reflect task intent. Extreme/streaming variants enforce stricter caps under poor precision or low coverage.

### Anti-exploit design

- sliding-window loop detection
- explicit penalty for repetitive degenerate patterns
- hard caps under pathological false-positive or low-coverage behavior
- report consistency checks to prevent post-hoc narrative fabrication

### Determinism and reproducibility

- task definitions are fixed and explicit
- event schedules are deterministic by task plus seed
- metrics and failure modes are emitted in structured outputs

This gives us an important property for research and debugging: when performance changes, we can replay the same scenario and inspect exactly where behavior diverged.

---

## Training Stack and Pipeline

We train with:

- OpenEnv for standardized interaction
- HF TRL (GRPO) + Unsloth as the primary training path
- Transformers + PEFT as a compatibility fallback via `--no-unsloth`

Practical workflow:

1. Run baseline inference sweeps on selected seeded tasks.
2. Train with GRPO against multi-signal reward.
3. Re-run evaluation on the same seeded slices.
4. Compare score, precision, coverage, consistency, and failure-mode distribution.

Using a fallback path mattered in practice: it kept experiments portable when hardware or kernel constraints made the accelerated route unavailable.

---

## What We Observed So Far

### Before RL

- Short tasks looked acceptable, but stability degraded as horizon and event complexity increased.
- Policies often over-indexed on local signals, then broke consistency later in the episode.
- Under pressure, behavior tended to split into two failure modes: under-flagging or precision collapse.

### After RL

- Action trajectories became more structured and repeatable.
- Recall improved without the same level of precision collapse.
- Report quality aligned better with actual episode history.
- Adaptation improved in delayed/sparse-reward settings.

### Current benchmark snapshot

- total logged runs: 54
- observed models: Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct, Mistral-Large-Instruct-2407
- top mean score in current logs: about 0.454

Final trained-vs-baseline deltas will be added after the complete eval sweep.

## Honest Limitations

- Time constraints limited the number of full training and evaluation rounds we could run.
- Model learning capability is still bounded: agents improve, but can plateau on complex cross-record reasoning and hard long-horizon edge cases.
- The current benchmark snapshot is informative but not yet a full ablation study across all tasks and seeds.

These limitations are not deal-breakers, but they are important context for interpreting current results.

---

## Closing

Auditrix is our attempt to push LLM RL beyond toy tasks into auditable operational reasoning.

The biggest lesson for us is that training is only half the problem. The harder half is environment design: making tasks realistic enough to matter, difficult enough to resist shortcut gaming, and structured enough to grade objectively.

If this direction resonates, we would love to collaborate with other teams building professional-task RL benchmarks.
