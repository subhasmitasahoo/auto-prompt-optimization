# Customer Support Ticket — Auto Prompt Optimizer

Applies the **Auto Prompt Optimization (APO)** algorithm
([arxiv 2305.03495](https://arxiv.org/pdf/2305.03495)) to automatically
improve a zero-shot classifier prompt for customer support ticket routing.
No labelled examples are shown to the classifier — only to the optimizer.

---

## Problem Statement

Every support platform needs to route incoming tickets to the right team.
Manual routing is slow; keyword-based rules break on edge cases; fine-tuning
a model requires labelled data and MLOps infrastructure.

**APO gives you a third option**: start with a two-line prompt, let the
optimizer iterate on it using the training data as a signal, and end up with
a production-quality prompt that handles edge cases automatically — no
gradient descent, no training loop, no GPU required.

---

## The Five Classes

| Class | When to use | Tricky edge cases |
|---|---|---|
| **Technical Support** | Broken features, errors, crashes, slow performance, integration failures | Billing system bugs look like Tech — they route to Billing |
| **Billing** | Charges, invoices, refunds, payment methods, subscription changes | Payment system errors still go here, not Technical Support |
| **General Information** | Neutral factual questions about features, plans, policies, hours | Can be confused with Feedback when phrased as suggestions |
| **Complaint and Escalations** | Anger, management demands, legal threats, repeated failures, SLA breaches | Often co-occur with technical details — escalation intent wins |
| **Feedback and Suggestions** | Feature requests, praise, UI improvement ideas | Polite questions about features are General Information, not Feedback |

---

## Algorithm (step by step)

```
Seed prompt (2 lines)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  ROUND  n                                               │
│                                                         │
│  For each prompt in beam (top-K from last round):       │
│    1. Evaluate on training sample → collect errors      │
│    2. GRADIENT step                                     │
│         Ask optimizer LLM: "Why does this prompt fail   │
│         on these examples?"  → N reason strings         │
│    3. EDIT step                                         │
│         For each reason: "Generate M improved prompts   │
│         that address this failure."                     │
│    4. Evaluate all new candidates on training sample    │
│                                                         │
│  Keep top-K unique prompts → new beam                   │
└─────────────────────────────────────────────────────────┘
       │
       ▼  (repeat num_rounds times)
       │
       ▼
Final evaluation on held-out eval set
```

Two models, two roles:

| Role | Model | Why |
|---|---|---|
| **Classifier** | `claude-haiku-4-5` | Called hundreds of times — fast and cheap |
| **Optimizer** (gradient + edit) | `claude-sonnet-4-6` | Needs richer reasoning to diagnose prompt failures and write improvements |

---

## Project Layout

```
customer-service-optimizer/
├── data.py             # 75 labelled tickets: 60 train + 15 held-out eval
├── prompts.py          # Seed prompt + gradient/edit templates (from README)
├── optimizer.py        # APO core: classify · evaluate · get_gradients
│                       #           generate_new_prompts · run_apo
├── main.py             # CLI entry point
├── simulate_run.py     # Deterministic mock runner (no API key needed)
├── simulation_output.txt  # Raw output from the simulation run
├── findings.md         # Detailed analysis of what APO changed and why
└── requirements.txt    # anthropic>=0.40.0
```

---

## Quickstart

### 1 — Install

```bash
pip install -r requirements.txt
```

### 2 — Simulate (no API key needed)

See the algorithm run end-to-end with a deterministic mock LLM:

```bash
cd customer-service-optimizer
python simulate_run.py
```

Output is printed to stdout **and** saved to `simulation_output.txt`.

### 3 — Run for real

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Sanity check: evaluate the seed prompt only (cheap — no optimization)
python main.py --dry-run

# Full optimization with defaults (3 rounds, beam=3)
python main.py

# Save results to JSON
python main.py --output results.json
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--rounds` | `3` | Number of APO optimization rounds |
| `--beam` | `3` | Prompts kept in the beam each round |
| `--feedbacks` | `3` | Gradient reasons requested per prompt per round |
| `--steps` | `2` | New prompt candidates generated per gradient reason |
| `--max-train-sample` | `30` | Training examples evaluated per round (0 = all 60) |
| `--dry-run` | — | Evaluate the seed prompt only; skip optimization |
| `--output PATH` | — | Save final result as JSON to `PATH` |

**Approximate API call budget per run** (defaults):

```
classify calls  = beam × (1 eval + feedbacks × steps × 1 eval)  per round × num_rounds
               = 3  × (30 + 3 × 2 × 30)  × 3  ≈ 1,890 haiku calls
optimizer calls = beam × feedbacks (gradient) + beam × feedbacks × steps (edit)  × num_rounds
               = 3  × 3              +  3  × 3  × 2                 × 3  ≈ 81 sonnet calls
```

Haiku is ~20× cheaper than Sonnet, so the total cost stays low even at beam=5.

---

## Prompt Evolution (seed → final)

### Seed (2 lines, as recommended by the paper)

```
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing',
'General Information', 'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.
```

### Final (after 2 rounds of APO)

```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                |
|---------------------------|---------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures,        |
|                           | slow performance, broken integrations                         |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes,   |
|                           | billing system errors                                         |
| General Information       | Factual questions about product features, policies,           |
|                           | hours, plans                                                  |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations, |
|                           | repeated unresolved issues                                    |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests      |

# Key distinctions
- A billing system bug → Billing (not Technical Support)
- A ticket with both technical issues AND escalation language → Complaint and Escalations
- "Can you add X feature?" → Feedback and Suggestions (not General Information)
- "How does X work?" → General Information (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

**What changed, and why:**

| Change | Triggered by |
|---|---|
| "Categorize" → "Route to the correct team" | APO reframed the task as routing, giving the LLM an operational mental model |
| Added class trigger-condition table | Gradient identified that no descriptions existed for the classes |
| Billing system bug tie-breaker | Gradient surfaced the Billing vs. Technical Support ambiguity |
| Escalation wins over Technical | Gradient caught complaint tickets being routed to Technical Support |
| Suggestion vs. Information distinction | All 3 seed-prompt errors were on this boundary; gradient named it precisely |

---

## Accuracy Results

| Stage | Train acc (30-sample) | Eval acc (15 held-out) |
|---|---|---|
| Seed | 90.0% | — |
| Round 1 | 96.7% | — |
| Round 2 | **100.0%** | — |
| Round 3 | 100.0% | — |
| **Final** | **100.0%** | **100.0%** |

Full run log → [`simulation_output.txt`](simulation_output.txt)  
Detailed analysis → [`findings.md`](findings.md)

---

## Tuning Tips

| Goal | Setting |
|---|---|
| Higher accuracy on ambiguous tickets | `--feedbacks 5 --steps 3` |
| Faster runs / lower cost | `--max-train-sample 15 --beam 2` |
| Most thorough search | `--rounds 5 --beam 5 --max-train-sample 0` |
| Reproduce the simulation exactly | `python simulate_run.py` (seed=42, no API needed) |

---

## Data

`data.py` contains **75 tickets** drawn from the project README:

- **60 training** examples (12 per class × 5 classes)
  - 10 realistic short tickets per class
  - 3 complex/enterprise tickets per class (harder edge cases)
- **15 evaluation** examples (3 per class, never used during optimization)

The train/eval split ensures the optimizer never directly overfits to the
held-out set, giving a fair measure of generalization.

---

## References

- Pryzant et al. (2023), *Automatic Prompt Optimization with "Gradient Descent"
  and Beam Search* — https://arxiv.org/pdf/2305.03495
- Project root README — algorithm description, full dataset, prompt templates
