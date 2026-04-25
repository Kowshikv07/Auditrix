# Auditrix: Teaching LLMs to Perform Real Compliance Audits with OpenEnv, TRL, and GRPO

Compliance workflows are one of those problems that look simple from the outside, but become complicated the moment you try to automate them safely.  
An auditor is not just checking a list. They are balancing recall and precision, handling exemptions, comparing across records, adapting to policy changes, and writing an accountable final report.

That is exactly what we wanted to train for.

`Auditrix` is our OpenEnv environment for reinforcement learning on professional compliance tasks.  
Instead of training an LLM to produce one static answer, we train it to act in a structured loop: inspect records, evaluate rules, gather evidence, flag violations, and generate an audit report under a step budget.

---

## Hackathon Problem Statement Alignment

Auditrix is positioned with:

- **Primary theme:** **#3.1 World Modeling (Professional Tasks)**
- **Secondary theme:** **#2 (Super) Long-Horizon Planning & Instruction Following**

### Primary: Theme #3.1 World Modeling (Professional Tasks)

We primarily target Theme #3.1 through:

- tool-based professional workflows (`inspect`, `apply_rule`, `flag`, `report`),
- partially observable state that updates over time,
- deterministic rule verification with evidence traces,
- and dynamic environment events that require belief updates and action re-planning.

This setup tests whether an agent can maintain a consistent world model while operating in a realistic enterprise compliance process, not just generate fluent text.

### Secondary: Theme #2 (Super) Long-Horizon Planning & Instruction Following

We cover Theme #2 through:

- long multi-step episodes with strict step budgets,
- sparse terminal-focused reward structure (plus low-magnitude shaping in streaming),
- mandatory strategy declaration via `prioritize_rules`,
- and delayed consequences from dynamic events (policy updates and record amendments).

In practice, this forces agents to plan early, allocate steps strategically, and recover from suboptimal earlier actions instead of relying on shallow one-step heuristics.

---

## The Problem We Targeted

Most benchmark-style RL environments are either:

- too toy-like to reflect enterprise constraints, or
- hard to verify objectively at scale.

Compliance auditing gives us the best of both worlds:

- **high practical relevance** (SOX, GDPR, HR controls),
- **objective verifiability** (deterministic rules and graders),
- **multi-step decision-making** (not one-shot QA),
- and **real failure costs** (false positives and missed violations).

Our core hypothesis: if we can train agents to handle this domain well, we are also improving broader capabilities in long-horizon planning, state tracking, and calibrated tool use.

---

## What Auditrix Simulates

Auditrix models a professional audit lifecycle in an OpenEnv-compatible interface.

### Agent action space

The agent can:

- `inspect_record`
- `apply_rule`
- `request_evidence`
- `flag_violation`
- `retract_flag`
- `mark_compliant`
- `prioritize_rules`
- `generate_report`

This keeps behavior explicit and auditable. No hidden shortcuts.

### Task suite

We include 8 tasks across increasing difficulty:

- **Easy/Medium**: deterministic starter audits for bootstrapping behavior
- **Domain audits**: SOX, GDPR, and data-integrity scenarios
- **Extreme stress test**: multi-rule, event-heavy audit with dynamic world updates
- **Streaming long-horizon**: 350 records, 400-step budget, sparse terminal-focused reward with low per-step shaping

### Rule coverage

Rules span labor law, payroll bands, data integrity, governance, and privacy:

- minors/intern overtime limits,
- salary-band validation,
- duplicate IDs,
- expired contracts,
- background checks,
- compliance training,
- GDPR consent,
- missing mandatory fields,
- manager reference integrity.

Depending on task, active rules are selected from R1-R11.

---

## Why This Is Challenging for an LLM Agent

Auditrix is intentionally built to stress agent weaknesses that matter in production:

1. **Exemption logic**  
   The best action is often to *not* flag a rule.

2. **Cross-record reasoning**  
   Some violations are only visible when comparing multiple records.

3. **Dynamic events**  
   Policy thresholds can change mid-episode, records can be amended, and temporary outages can block access.

4. **Long-horizon planning**  
   A naive inspect-everything policy fails under strict step budgets.

5. **Report accountability**  
   Final reports are scored for consistency and completeness, not just presence.

---

## Reward Design: Multi-Signal by Construction

We avoided single-number reward design because it is fragile and easy to game.

Instead, we train on multiple independent reward signals, including:

- overall task score,
- severity-weighted detection,
- precision / false-positive behavior,
- coverage,
- efficiency,
- prioritization quality,
- deliberation quality (`request_evidence` usage),
- anti-exploit penalties.

This gives us two benefits:

- stronger training signal,
- better observability when behavior regresses or starts exploiting loopholes.

---

## Training Stack

Auditrix is trained with:

- **OpenEnv** for standardized environment interaction,
- **HF TRL (GRPO)** for policy optimization,
- **Unsloth** for efficiency on supported GPU setups,
- and a **HF Transformers + PEFT fallback** (`--no-unsloth`) for incompatible systems.

This made it straightforward to run fast experiments and still keep a reproducible baseline path.

---

## What We Observed

### Baseline (before RL)

Strong instruct models can perform reasonably on simpler audits, but we consistently see:

- missed multi-rule violations,
- low coverage on harder tasks,
- brittle behavior when dynamic events fire,
- and weaker report consistency.

### After RL fine-tuning

We see improvements in:

- structured, repeatable audit trajectories,
- higher useful violation recall without runaway false positives,
- better final report quality and internal consistency,
- more stable behavior under delayed/sparse reward conditions.

> Replace with your measured metrics from final run:
>
> - Baseline average score: **[X.XX]**
> - Trained average score: **[Y.YY]**
> - Relative improvement: **[+Z%]**
> - Biggest gain task: **[TASK_NAME, +N.NN]**

---

## Why This Matters Beyond the Hackathon

Auditrix is a practical testbed for training **auditable decision agents**, not just fluent text generators.

Potential downstream uses:

- compliance copilot evaluation,
- internal control monitoring,
- policy-constrained workflow automation,
- trustworthy AI benchmarking for enterprise operations.

More broadly, this project shows how RL can improve domain behavior when:

- actions are explicit,
- verification is objective,
- and reward design reflects real operational priorities.

---

## Reproducibility and Artifacts

- Environment (HF Space): **[PASTE_SPACE_URL]**
- Code repository: **[PASTE_REPO_URL]**
- Training notebook: **[PASTE_COLAB_OR_IPYNB_URL]**
- Trained model/adapters: **[PASTE_MODEL_URL]**
- Training plots (reward/loss/before-after): **[PASTE_ARTIFACT_URL]**
- Demo video or short presentation: **[PASTE_VIDEO_OR_SLIDES_URL]**

---

## Suggested 2-Minute Demo Flow

For quick review, we recommend this sequence:

1. Run baseline on one medium and one hard task.
2. Show reward/grader outputs.
3. Run trained model on same seeded tasks.
4. Compare score, precision, coverage, and report quality.
5. Highlight anti-exploit behavior and dynamic event handling.

This makes the learning effect obvious in under two minutes.

---

## Closing

Auditrix is our attempt to move LLM RL from toy tasks toward real operational reasoning: constrained actions, delayed outcomes, and auditable decisions.

If you are building similar professional-task RL environments, we would love to compare methods and results.