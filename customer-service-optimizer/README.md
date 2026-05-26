# Customer Support Ticket — Auto Prompt Optimizer

Applies the **Auto Prompt Optimization (APO)** algorithm
([arxiv 2305.03495](https://arxiv.org/pdf/2305.03495)) to automatically
improve a zero-shot classifier prompt for customer support ticket routing.

---

## Task

Given a free-text support ticket, classify it into one of **5 categories**:

| Class | Example ticket |
|---|---|
| Technical Support | "The app keeps crashing on iOS 17 after the last update." |
| Billing | "I was charged twice for my monthly subscription." |
| General Information | "Do you offer student discounts on premium plans?" |
| Complaint and Escalations | "This is the third time I've reported this issue — escalate now!" |
| Feedback and Suggestions | "Dark mode would make the dashboard so much easier on the eyes." |

---

## How it works

```
Round 0 ── evaluate seed prompt ──────────────────────────────────────────┐
                                                                           │
For each round:                                                            │
  For each prompt in beam:                                                 │
    1. Evaluate on training sample  → collect wrong predictions            │
    2. Gradient step  → ask LLM for N reasons why the prompt fails         │
    3. Edit step      → generate M improved prompts per reason             │
    4. Evaluate all candidates                                             │
  Keep top-K prompts  →  new beam                                          │
                                                                           │
Final step ── evaluate best prompt on held-out eval set ──────────────────┘
```

Two models are used:
- **`claude-haiku-4-5`** — bulk classification (fast & cheap, called ~hundreds of times)
- **`claude-sonnet-4-6`** — gradient & edit steps (smarter reasoning for meta-prompting)

---

## Project layout

```
customer-service-optimizer/
├── data.py          # 75 labeled tickets (60 train + 15 eval, 5 classes × 3 eval each)
├── prompts.py       # Seed prompt + gradient / edit templates
├── optimizer.py     # APO core: classify, evaluate, get_gradients, generate_new_prompts, run_apo
├── main.py          # CLI entry point
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Check how the seed prompt performs (no API cost for optimization)
python main.py --dry-run

# 4. Run full optimization (3 rounds, beam=3 — good starting point)
python main.py

# 5. Save results to JSON
python main.py --output results.json
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--rounds` | 3 | Number of APO optimization rounds |
| `--beam` | 3 | Prompts kept in beam each round |
| `--feedbacks` | 3 | Gradient reasons requested per prompt |
| `--steps` | 2 | New prompt candidates generated per gradient reason |
| `--max-train-sample` | 30 | Training examples per evaluation (0 = all) |
| `--dry-run` | — | Evaluate seed prompt only, no optimization |
| `--output` | — | Path to save JSON results |

---

## Expected output (abridged)

```
============================================================
  AUTO PROMPT OPTIMIZATION — Customer Support Classifier
============================================================

📌 Initial prompt:
# Task
Categorize the customer support ticket.
...

📊 Seed accuracy on train sample: 72.0%  (8 errors)

────────────────────────────────────────────────────────────
  Round 1 / 3
────────────────────────────────────────────────────────────
  ▸ Beam prompt #1  (train acc = 72.0%)
    🔍 Getting 3 gradient feedback(s)…
    ✏️  Generating 6 candidate prompt(s)…
      Gradient 1, candidate 1: train acc = 80.0%
      ...

============================================================
  OPTIMIZATION COMPLETE
============================================================

🎯 Final prompt:
...

📊 Train accuracy : 90.0%
📊 Eval  accuracy : 86.7%
```

---

## Tuning tips

- **More rounds** → better prompts, more API calls
- **Larger beam** → explores more paths, higher cost
- **More feedbacks** → richer gradient signals, but diminishing returns past 5
- **`--max-train-sample 0`** → use all 60 training examples per round (most accurate but slowest)
