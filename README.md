# Auto Prompt Optimization (APO)

Implementation of [*Automatic Prompt Optimization with "Gradient Descent" and Beam Search*](https://arxiv.org/pdf/2305.03495) (Pryzant et al., 2023).

APO automatically improves a zero-shot classifier prompt by iterating over training errors — no gradient descent, no fine-tuning, no GPU required. The optimizer **is** the LLM.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Prompt Templates](#prompt-templates)
3. [Implementation](#implementation)
4. [Data Format](#data-format)
5. [Results](#results)
6. [Exercises](#exercises)
7. [Contributing](#contributing)

---

## How It Works

```
Step 1 — Start with a minimal seed prompt (2 lines)
           Just the task description + class names. No descriptions.

Step 2 — Evaluate
           Run the prompt on a labelled training sample.
           Collect every misclassified example.

Step 3 — Gradient step
           Show the wrong examples to a capable LLM and ask:
           "Why did this prompt fail on these examples?"
           → Returns N natural-language reasons (the "gradients").

Step 4 — Edit step
           For each gradient reason, ask the LLM:
           "Rewrite the prompt to fix this failure."
           → Returns M improved prompt candidates.

Step 5 — Evaluate all candidates
           Score every new prompt on the training sample.
           Keep the top-K (beam search).

Step 6 — Repeat from Step 2
           Run for num_rounds. Select the best prompt.
           Evaluate once on the held-out eval set.
```

### Key design choices

| Choice | Reasoning |
|---|---|
| **Minimal seed prompt** | Gives the optimizer maximum room to improve. Starting with a rich prompt obscures which additions are actually useful. |
| **Beam search over greedy** | A single gradient direction can produce a candidate that regresses. Keeping K prompts in parallel absorbs bad gradients without losing progress. |
| **Two-model strategy** | Use a fast/cheap model (e.g. Haiku) for the hundreds of classification evaluations; use a capable model (e.g. Sonnet) only for gradient and edit steps. |
| **Natural-language gradients** | Unlike numeric gradients, these are human-readable — each one is a legible explanation of a failure mode. |
| **No few-shot examples in classifier** | APO optimizes the prompt itself; injecting examples would conflate prompt quality with example selection. |

### Algorithm parameters

| Parameter | What it controls |
|---|---|
| `num_rounds` | How many optimization iterations to run |
| `beam_size` | Number of top prompts kept between rounds |
| `num_feedbacks` | Gradient reasons requested per prompt per round |
| `steps_per_gradient` | New prompt candidates generated per gradient reason |
| `max_train_sample` | How many training examples evaluated per round (cap for speed) |

**Candidates generated per round** = `beam_size × num_feedbacks × steps_per_gradient`

With defaults (beam=3, feedbacks=3, steps=2): **18 candidates per round**.

---

## Prompt Templates

### Seed prompt (start here)

```
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing',
'General Information', 'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.
```

Deliberately two lines. No class descriptions — APO will write them.

---

### Gradient prompt template

Sent to the **optimizer model** with the wrong examples filled in.
Elicits natural-language reasons for the prompt's failures.

```
I'm trying to write a zero-shot classifier prompt.
My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
Wrap each reason with <START> and <END>
```

**Parameters:**

| Placeholder | Value |
|---|---|
| `{prompt}` | The current best prompt text |
| `{error_string}` | Numbered list of misclassified examples: text, expected label, predicted label |
| `{num_feedbacks}` | How many gradient reasons to request (default: 3) |

**Output format:** One or more `<START>reason<END>` blocks. Parse with `re.findall(r"<START>(.*?)<END>", response, re.DOTALL)`.

---

### Edit prompt template

Sent to the **optimizer model** once per gradient reason.
Generates improved prompt candidates that address the identified failure.

```
I'm trying to write a zero-shot classifier.
My current prompt is:
"{prompt}"

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that {gradient}

Based on the above information, write {steps_per_gradient} different improved prompts.
To handle the losses, consider making the descriptions of each class in the improved prompts.
Each prompt should be wrapped with <START> and <END>.
The {steps_per_gradient} new prompts are:
```

**Parameters:**

| Placeholder | Value |
|---|---|
| `{prompt}` | The current best prompt text |
| `{error_str}` | Same misclassified examples as above |
| `{gradient}` | One gradient reason string (from the gradient step) |
| `{steps_per_gradient}` | How many improved prompts to generate (default: 2) |

**Output format:** One or more `<START>improved_prompt<END>` blocks. Parse identically to the gradient step.

---

## Implementation

The reference implementation is in [`customer-service-optimizer/`](customer-service-optimizer/).
It applies APO to customer support ticket routing across 5 classes.

```
customer-service-optimizer/
├── data.py              # 75 labelled tickets: 60 train + 15 held-out eval
├── prompts.py           # Seed prompt + gradient/edit templates
├── optimizer.py         # APO core functions
│   ├── classify()       #   Run one classification call
│   ├── evaluate()       #   Score a prompt on a dataset → (accuracy, errors)
│   ├── get_gradients()  #   Gradient step → list of reason strings
│   ├── generate_new_prompts()  # Edit step → list of new prompt strings
│   └── run_apo()        #   Full optimization loop
├── main.py              # CLI entry point (--rounds, --beam, --dry-run, ...)
├── simulate_run.py      # Deterministic mock runner — no API key needed
├── simulation_output.txt   # Raw output from the simulation
├── sample_outputs.md    # Annotated output: all gradients + edit prompts
├── findings.md          # Analysis: what APO changed and why
├── medium_post.md       # Write-up of the experiment
└── requirements.txt     # anthropic>=0.40.0
```

### Core functions (`optimizer.py`)

#### `classify(prompt, ticket) → str`

Sends a single classification request to the classifier model. Includes response parsing: tries exact match, then substring match, then returns raw response.

```python
CLASSIFY_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap

def classify(prompt: str, ticket: str) -> str:
    message = client.messages.create(
        model=CLASSIFY_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": f"{prompt}\n\nTicket: {ticket}"}],
    )
    return parse_prediction(message.content[0].text)
```

#### `evaluate(prompt, data) → (float, list[dict])`

Evaluates a prompt against a full dataset. Returns accuracy and a list of error dicts (`text`, `expected`, `predicted`).

```python
def evaluate(prompt: str, data: List[Example]) -> Tuple[float, List[Dict]]:
    correct, errors = 0, []
    for example in data:
        predicted = classify(prompt, example.text)
        if predicted.lower() == example.label.lower():
            correct += 1
        else:
            errors.append({"text": example.text,
                           "expected": example.label,
                           "predicted": predicted})
    return correct / len(data), errors
```

#### `get_gradients(prompt, errors, num_feedbacks) → list[str]`

Sends the gradient prompt to the optimizer model. Parses `<START>…<END>` blocks from the response.

```python
OPTIMIZE_MODEL = "claude-sonnet-4-6"

def get_gradients(prompt, errors, num_feedbacks=3) -> List[str]:
    gradient_query = GRADIENT_PROMPT_TEMPLATE.format(
        prompt=prompt,
        error_string=format_errors(errors),
        num_feedbacks=num_feedbacks,
    )
    response = client.messages.create(model=OPTIMIZE_MODEL, ...).content[0].text
    return [g.strip() for g in re.findall(r"<START>(.*?)<END>", response, re.DOTALL)]
```

#### `generate_new_prompts(prompt, errors, gradient, steps_per_gradient) → list[str]`

Sends the edit prompt to the optimizer model. Returns up to `steps_per_gradient` new prompt strings.

#### `run_apo(initial_prompt, train_data, eval_data, ...) → dict`

The full loop. Returns `best_prompt`, `best_train_accuracy`, `best_eval_accuracy`, and `history`.

```python
result = run_apo(
    initial_prompt     = INITIAL_PROMPT,
    train_data         = TRAINING_DATA,
    eval_data          = EVAL_DATA,
    num_rounds         = 3,
    beam_size          = 3,
    steps_per_gradient = 2,
    num_feedbacks      = 3,
    max_train_sample   = 30,
)
```

### Running it

```bash
cd customer-service-optimizer
pip install -r requirements.txt

# No API key — run the deterministic simulation
python simulate_run.py

# With API key — dry run (evaluate seed only, no optimization)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --dry-run

# Full run
python main.py --rounds 3 --beam 3

# Save results to JSON
python main.py --output results.json
```

---

## Data Format

Training and evaluation data are lists of `Example` dataclass instances:

```python
@dataclass
class Example:
    text: str    # the support ticket text
    label: str   # one of the five class names
```

```python
TRAINING_DATA = [
    # Technical Support
    Example("I can't log into my account, it keeps saying 'invalid credentials'...", "Technical Support"),
    Example("The app keeps crashing whenever I try to upload a photo...", "Technical Support"),
    # ... 13 more Technical Support examples

    # Billing
    Example("I was charged twice for my monthly subscription...", "Billing"),
    # ... 14 more Billing examples

    # General Information
    Example("What are your business hours during the holiday season?", "General Information"),
    # ... 14 more General Information examples

    # Complaint and Escalations
    Example("I've been trying to resolve this issue for weeks...", "Complaint and Escalations"),
    # ... 14 more Complaint and Escalations examples

    # Feedback and Suggestions
    Example("It would be great if you could add a dark mode option...", "Feedback and Suggestions"),
    # ... 14 more Feedback and Suggestions examples
]
```

The full dataset (75 examples) is in [`customer-service-optimizer/data.py`](customer-service-optimizer/data.py).

**Data split:** 60 training / 15 held-out evaluation (3 per class for eval, never used during optimization).

**Ticket complexity:** Each class contains both short simple tickets (single-issue, consumer-level) and long complex ones (multi-issue, enterprise-level). Complex tickets are the hard cases APO is especially good at, because they often span multiple class boundaries.

---

## Results

Run on the customer support classifier (3 rounds, beam=3, 30-ticket sample):

| Stage | Train acc | Eval acc | Notes |
|---|---|---|---|
| Seed | 90.0% | — | All errors on General Info ↔ Feedback boundary |
| Round 1 | 96.7% | — | Gradient fixed intent distinction |
| Round 2 | **100.0%** | — | Convergence |
| Round 3 | 100.0% | — | No candidate improved on best; early-stop signal |
| **Final** | **100.0%** | **100.0%** | 0 errors on 15 held-out tickets |

**What the optimizer added** (none of this was in the seed):

- Explicit trigger-condition table per class
- Task reframed from "categorize" → "route to the correct **team**"
- Billing system bugs → Billing (not Technical Support)
- Escalation intent overrides Technical Support classification
- Suggestions are opinions; information requests are neutral queries

Full output with all gradient strings and generated prompts: [`sample_outputs.md`](customer-service-optimizer/sample_outputs.md)

---

## Exercises

These are open tasks for anyone wanting to experiment with APO, build on the implementation, or contribute back to the repo. They range from quick experiments to full new modules.

Tasks marked **[contribute]** are good PR candidates — see [Contributing](#contributing) for guidelines.

---

### Beginner

**E1 — Try a different seed prompt** *(no code changes)*
Run `main.py --dry-run` with a more detailed seed: add class descriptions yourself before running APO. Does APO still improve it? Does it converge faster or reach a higher ceiling?

**E2 — Vary the beam size** *(CLI only)*
Compare `--beam 1` (greedy) vs `--beam 3` (default) vs `--beam 5` across 3 rounds. Does a wider beam consistently find better prompts, or do returns diminish quickly?

**E3 — Vary the number of feedbacks** *(CLI only)*
Try `--feedbacks 1`, `--feedbacks 3`, `--feedbacks 5`. Plot the best candidate score per round. Do more gradient reasons produce meaningfully different prompts, or do they converge to the same insights?

---

### Intermediate

**E4 — Add 15 more training examples per class** **[contribute]**
The current dataset has 15 examples per class (12 train + 3 eval). Add 15 more to `data.py` — your own original tickets, not copies of existing ones. Focus on edge cases: tickets that could belong to two classes, highly technical tickets, very short tickets, non-native English speaker style. Run APO with `--max-train-sample 0` and compare results.

*Guidelines for new data: see [Contributing → Adding training examples](#adding-training-examples).*

**E5 — Implement early stopping** **[contribute]**
Add a flag `--early-stop` to `main.py` and `run_apo()`. If no candidate in a round beats the current best prompt accuracy, stop and return. Report how many rounds were saved vs. the default run.

**E6 — Add a new domain** **[contribute]**
Create a new subdirectory (e.g. `ecommerce-optimizer/` or `hr-tickets-optimizer/`) following the same module structure as `customer-service-optimizer/`. Define 4–6 classes, write 50+ labelled examples, and run APO on it. Document what the optimizer changed and why in a `findings.md`.

*See [Contributing → Adding a new domain](#adding-a-new-domain) for structure requirements.*

**E7 — Log all beam prompts per round**
Modify `run_apo()` to return (and optionally save) all prompts in the beam at each round, not just the best. Visualise how the beam converges. Do all K prompts end up very similar by round 3, or does diversity remain?

**E8 — Test on out-of-distribution tickets**
Write 10 tickets that deliberately don't fit any class cleanly (e.g. a ticket that is simultaneously a technical bug report, a billing complaint, and an escalation). Run both the seed and the final optimized prompt on them. Does APO's reframing help or hurt on truly ambiguous inputs?

---

### Advanced

**E9 — Multi-label APO** **[contribute]**
The current setup is single-label classification (one class per ticket). Some tickets genuinely span two categories. Modify the classifier to output a primary and optional secondary class, update the evaluation metric accordingly (e.g. partial credit), and re-run APO. Does the gradient step naturally suggest multi-label improvements?

**E10 — APO with a smaller model**
Replace `OPTIMIZE_MODEL` (currently Sonnet) with Haiku for the gradient and edit steps too. Does the optimizer still produce useful gradient feedback with a smaller model? How does accuracy at convergence compare? Useful for cost-sensitive deployments.

**E11 — Implement confidence-weighted evaluation**
Instead of binary correct/wrong, weight errors by model confidence (using logprobs if available, or asking the model to rate its own certainty). Feed only high-confidence wrong predictions to the gradient step — the hypothesis is that low-confidence misclassifications are inherently ambiguous and produce noisy gradients.

**E12 — APO for a regression task**
The paper focuses on classification. Adapt the optimizer for a scoring task — e.g. ticket urgency on a 1–5 scale. The gradient step needs to reason about *direction* of error (scored too high vs too low), not just wrong/right. Redefine the error string format and evaluate with MAE instead of accuracy.

---

## Contributing

Contributions are welcome. The most useful contributions are:

1. **New training examples** for the existing customer support dataset
2. **New domain modules** (new task + dataset + findings)
3. **Algorithm improvements** (early stopping, confidence weighting, etc.)
4. **Bug fixes and clarity improvements** to existing code

---

### General guidelines

- All code should be runnable with `python simulate_run.py` (no API key) for reviewers who don't have access
- If you add a live API run, also add or update a `simulate_run.py` with a deterministic mock
- No new dependencies beyond `anthropic` unless essential and justified in the PR description
- Follow the existing file naming convention: `data.py`, `prompts.py`, `optimizer.py`, `main.py`

---

### Adding training examples

When adding to `data.py`:

```python
# Good — original, specific, plausible
Example("My two-factor authentication stopped working after I got a new phone number. I can still receive SMS but the app says the code is invalid.", "Technical Support")

# Avoid — vague, too short to be informative
Example("The thing isn't working.", "Technical Support")

# Avoid — near-duplicate of an existing example
Example("I cannot log into my account due to invalid password.", "Technical Support")  # too similar to existing
```

**Checklist for new examples:**
- [ ] Original text — not copied or lightly paraphrased from existing examples
- [ ] Clear label — if you're unsure which class it belongs to, it's probably a good edge-case example; add a comment explaining your reasoning
- [ ] Realistic language — write as a real user would, not as a data labeller
- [ ] Mix of lengths — include both short (1–2 sentences) and long (3–5 sentences) tickets
- [ ] No PII — no real names, emails, account numbers, or company names
- [ ] Add to both `TRAINING_DATA` (for optimization) and consider adding 1 per class to `EVAL_DATA` (held-out)

---

### Adding a new domain

Create a subdirectory named `<domain>-optimizer/` with this structure:

```
<domain>-optimizer/
├── data.py          # Example dataclass + TRAINING_DATA + EVAL_DATA + CLASSES list
├── prompts.py       # INITIAL_PROMPT + GRADIENT_PROMPT_TEMPLATE + EDIT_PROMPT_TEMPLATE
├── optimizer.py     # Copy from customer-service-optimizer/ — update CLASSES reference
├── main.py          # Copy from customer-service-optimizer/ — update description string
├── simulate_run.py  # Mock runner with domain-specific confusion matrix
├── requirements.txt # anthropic>=0.40.0
├── findings.md      # What APO changed, accuracy progression, interesting observations
└── README.md        # Problem statement, classes, quickstart, results
```

**Minimum data requirements:**
- At least 4 classes
- At least 10 training examples per class (15+ recommended)
- At least 3 evaluation examples per class (held-out, never used in optimization)
- At least one "complex" example per class (multi-sentence, edge-case, or ambiguous)

**Domain ideas:**
- E-commerce order issues (Shipping, Returns, Payment, Product Quality, Account)
- HR ticket triage (Leave Request, IT Access, Payroll, Policy Question, Complaint)
- Developer support (Bug Report, Feature Request, Documentation, Integration Help, Account)
- Healthcare patient portal (Appointment, Prescription, Billing, Test Results, General)
- Legal intake (Contract Review, IP, Employment, Litigation, General Inquiry)

---

### Submitting a PR

1. Fork the repo and create a branch named `<your-name>/<short-description>`
2. Make your changes — code + data + `simulate_run.py` + `findings.md`
3. Run `python simulate_run.py` and confirm it exits cleanly
4. Open a PR with:
   - A one-sentence description of what you added or changed
   - Accuracy before/after (if you modified the algorithm or data)
   - Any interesting observations from your run

---

## References

- Pryzant et al. (2023) — *Automatic Prompt Optimization with "Gradient Descent" and Beam Search* — https://arxiv.org/pdf/2305.03495
- Reference implementation write-up — [`customer-service-optimizer/medium_post.md`](customer-service-optimizer/medium_post.md)
- Full run output with gradient strings — [`customer-service-optimizer/sample_outputs.md`](customer-service-optimizer/sample_outputs.md)
- Detailed findings — [`customer-service-optimizer/findings.md`](customer-service-optimizer/findings.md)
