# I Let an AI Rewrite Its Own Prompt — And It Found Edge Cases I Never Thought Of

## Auto Prompt Optimization: from a 2-line seed to a production-ready classifier in 2 rounds

---

Every ML engineer has been there. You spend an afternoon writing a carefully crafted prompt, test it on a dozen examples, feel good about it — and then watch it silently misroute tickets for weeks because it can't tell the difference between someone asking *about* a feature and someone *requesting* that feature.

What if you could skip that loop entirely? What if the model could figure out what's wrong with its own prompt — and fix it?

That's the core idea behind **Auto Prompt Optimization (APO)**, a paper from Stanford ([arxiv 2305.03495](https://arxiv.org/pdf/2305.03495)) that treats prompt engineering as an optimization problem and solves it with gradient descent — except the "gradients" are natural language, and the optimizer is the model itself.

I built an implementation of it for customer support ticket classification. Here's what happened.

---

## The Problem

Support platforms route thousands of tickets daily across teams: Technical Support, Billing, General Information, Complaint and Escalations, Feedback and Suggestions. Get the routing wrong and a furious customer who wants to escalate ends up in a queue where a billing agent tells them to try clearing their cache.

Manual routing is slow. Keyword rules break. Fine-tuning requires data infrastructure. A well-crafted LLM prompt is the pragmatic middle ground — but writing that prompt is harder than it looks.

---

## The Algorithm in Plain English

APO works in three steps, repeated across multiple rounds:

**1. Evaluate** — run your current prompt on a labelled training set and collect the examples it gets wrong.

**2. Gradient step** — show those wrong examples to a capable model and ask: *"Why did the prompt fail on these?"* The model returns natural-language reasons — these are the "gradients."

**3. Edit step** — for each gradient reason, ask the model to generate improved prompt versions that address the identified failure.

Keep the top-K prompts (beam search), repeat. After a few rounds you have a prompt that has diagnosed its own blind spots and rewritten itself to fix them.

```
Seed prompt (2 lines)
       │
       ▼
  ┌────────────────────────────────────────┐
  │  For each prompt in beam:              │
  │    1. Evaluate → collect errors        │
  │    2. Ask model: "Why did you fail?"   │
  │    3. Ask model: "Fix the prompt"      │
  │    4. Evaluate all new candidates      │
  │  Keep top-K → new beam                 │
  └────────────────────────────────────────┘
       │
       ▼  (repeat N rounds)
  Final eval on held-out set
```

The seed prompt is intentionally minimal — just two lines. The paper recommends starting simple so the optimizer has room to work.

---

## What I Built

The module lives in `customer-service-optimizer/` and has four main files:

- **`data.py`** — 75 labelled tickets: 60 for training, 15 held-out for final evaluation. Five classes, including both short simple tickets and long complex enterprise tickets.
- **`prompts.py`** — the seed prompt and the gradient/edit templates, taken directly from the paper.
- **`optimizer.py`** — the APO engine: `classify`, `evaluate`, `get_gradients`, `generate_new_prompts`, and the main `run_apo` loop.
- **`main.py`** — a CLI with configurable rounds, beam size, and a `--dry-run` mode.

One design decision worth noting: **two models, two roles.**

```python
CLASSIFY_MODEL  = "claude-haiku-4-5"   # fast, cheap — called ~1,890 times per run
OPTIMIZE_MODEL  = "claude-sonnet-4-6"  # smarter — gradient + edit steps (~81 calls)
```

Classification is a bulk operation — you're evaluating every prompt candidate against every training example every round. Haiku handles that affordably. The gradient and edit steps require genuine reasoning about *why* a prompt fails and *how* to fix it — that's where you want Sonnet.

---

## The Seed Prompt

Following the paper's recommendation, I started with two lines:

```
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing',
'General Information', 'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.
```

No descriptions. No examples. No guidance on edge cases. Just the task and the class names.

---

## Round 0: The Seed Is Surprisingly Good — and Blind in One Specific Spot

Seed accuracy on the 30-ticket training sample: **90%**. Three errors. All three were the same mistake:

| True label | Predicted | Ticket |
|---|---|---|
| General Information | Feedback and Suggestions | "Can you tell me more about your enterprise plan features?" |
| Feedback and Suggestions | General Information | "Your onboarding process was smooth, but a video tutorial would be helpful." |
| Feedback and Suggestions | General Information | "Consider adding bulk upload functionality — it would save us a lot of time." |

The model nailed Technical Support, Billing, and Complaint tickets. But it couldn't distinguish between someone *asking about* a feature (General Information) and someone *suggesting* a feature (Feedback and Suggestions). Both classes use polite, non-urgent language. Both often mention product features. Without descriptions, the model defaulted to surface-level word matching.

---

## Round 1: The Gradient Step Names the Problem Precisely

Here's what the gradient step returned — without any human involvement:

> *"it treats all polite, positive messages as Feedback and Suggestions, but messages asking about features or plans are General Information requests even if they're phrased positively."*

> *"it confuses General Information queries with Feedback and Suggestions when users phrase questions as implicit suggestions (e.g. 'Can you add X?'). Feedback implies the user is offering an opinion or improvement idea, while General Information is a neutral factual query."*

That is a precise, correct diagnosis. It identified the exact linguistic boundary the seed prompt couldn't see: **intent** — neutral inquiry vs. opinion/suggestion. No human wrote that. The model was shown the three wrong examples and figured it out.

The edit step produced six candidate prompts. Best performers (96.7% train accuracy):

```
# Task
You are a customer support ticket router. Assign each ticket to the
single most appropriate class below.

# Routing rules
1. If the ticket reports a broken feature, error, or technical malfunction
   → Technical Support
2. If the ticket is about money: charges, invoices, refunds, payment, pricing
   → Billing (even if a billing system has a bug, route to Billing not Technical Support)
3. If the ticket asks a neutral question about the product, features, or policies
   → General Information
4. If the ticket contains threats, demands for management, legal language, or
   expresses severe dissatisfaction after repeated failures → Complaint and Escalations
   (prioritise this over Technical Support if BOTH apply)
5. If the ticket praises the product, suggests an improvement, or requests a new feature
   → Feedback and Suggestions
   (distinguish from General Information: suggestions are opinions, not factual queries)
```

Notice what appeared: an explicit rule that **Billing beats Technical Support** when a payment system has a technical error, and that **escalation intent beats Technical Support** when a ticket contains both. Neither of those rules were in the seed. The gradient step surfaced real structural ambiguities in the dataset — even ones not in the three errors it was shown.

One candidate actually **regressed** — scoring 86.7%, below the seed. The gradient that produced it over-generalised and introduced new confusions. Beam search discarded it naturally. This is why beam size matters: a single-candidate approach has no safety net against bad gradients.

---

## Round 2: Convergence — 100% Train Accuracy

Round 2 had three beam prompts, all at 96.7%. One of the candidates hit **100%** for the first time:

```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                        |
|---------------------------|-------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures,|
|                           | slow performance, broken integrations                 |
| Billing                   | Payments, invoices, refunds, pricing, subscription    |
|                           | changes, billing system errors                        |
| General Information       | Factual questions about product features, policies,   |
|                           | hours, plans                                          |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA     |
|                           | violations, repeated unresolved issues                |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement       |
|                           | requests                                              |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language
  → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

Look at what changed:

| Dimension | Seed | Final |
|---|---|---|
| Task framing | "Categorize" | "Route to the correct **team**" |
| Class descriptions | None | Full trigger-condition table |
| Tie-breaker rules | None | 4 explicit edge-case rules |
| Escalation detection | Absent | "legal threats, SLA violations, repeated issues" |
| Info vs Suggestion | Indistinguishable | "suggestions are opinions, not factual queries" |
| Billing/Tech boundary | Absent | "billing system errors → Billing" |

The optimizer didn't just fill in descriptions — it **reframed the task**. "Categorize" became "route to a team." That's a subtle but powerful shift: it gives the model a mental model anchored in real-world workflow (who owns the next action?) rather than an abstract linguistic label.

---

## Round 3: The Early Stopping Signal

Round 3 generated 18 new candidates. Eleven of them scored 100%. Zero of them beat the existing best. The beam simply held.

This reveals a practical rule: **if no candidate improves on the best in a round, stop.** In this run, adding `early_stopping=True` would have saved 33% of API calls with no accuracy loss.

---

## Final Evaluation on the Held-Out Set

The 15 held-out tickets — never touched during optimization — scored **100%**. Zero errors.

```
Seed    : 90.0%  (3 errors)
Round 1 : 96.7%  (1 error)
Round 2 : 100.0% (0 errors)  ← convergence
Round 3 : 100.0% (0 errors)  ← maintenance
Eval    : 100.0% (0 errors on held-out)
```

---

## The Interesting Parts

**The gradient step is a free linguistic error analysis.** You don't need to manually review misclassifications and write rules. The model does it. In two rounds it identified: the General Info / Feedback boundary, the Complaint / Technical ambiguity, the Billing / Technical edge case for payment system bugs, and the problem of composite enterprise tickets that span multiple categories. That's the kind of insight a senior support engineer develops after months of production experience.

**Not all gradients are useful.** In Round 1, one gradient reason produced a candidate that scored *worse* than the seed. The gradient was technically correct (it identified a real structural issue) but the edit step over-corrected and introduced new confusions elsewhere. Beam search is the safety net here — it absorbs bad directions without penalising the whole run.

**The framing shift matters.** The final prompt uses "route to a team" instead of "categorize." This wasn't in the gradient feedback or the edit instructions — it emerged spontaneously. It likely helps because it activates a different reasoning path in the model: instead of "what label fits this text," it's "which team would own the next action?" For tickets that span two categories, the latter question has a clearer answer.

**APO generalises beyond factual tasks.** The original paper demonstrates APO on classification tasks where there's an objectively correct answer. Customer support routing has domain-specific rules (billing team owns billing-system bugs even if the bug is technical in nature) that can't be inferred from text alone. APO surfaced the *need* for these rules even when it couldn't know the correct answer — the gradient identified the structural ambiguity, and a domain expert would confirm the rule.

---

## Running It Yourself

```bash
git clone https://github.com/subhasmitasahoo/auto-prompt-optimization
cd auto-prompt-optimization/customer-service-optimizer

pip install -r requirements.txt

# No API key? Run the deterministic simulation:
python simulate_run.py

# With an API key — full live run:
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --dry-run          # just evaluate the seed
python main.py                    # 3 rounds, beam=3
python main.py --rounds 5 --beam 5 --output results.json
```

The simulation (`simulate_run.py`) runs the full APO loop with a deterministic mock LLM — no API key needed. Every gradient string and generated prompt is printed inline so you can see exactly how the algorithm progresses.

---

## What I'd Do Differently in Production

**More training data.** 30 tickets per round is enough to see the algorithm work, but real production evaluation needs at least 200–300 examples per class to get stable accuracy estimates. With small samples, one awkward ticket can swing accuracy by 3%.

**Periodic re-optimization.** Ticket language drifts. New product features create new support patterns. A quarterly re-run of APO on fresh tickets — with the previous best prompt as the new seed — would keep the classifier current without any manual intervention.

**Version every generated prompt.** The beam discards losers, but discarded prompts are useful. A prompt that scored 86.7% might be the right starting point for a different task. Logging all candidates to a database costs almost nothing and pays off eventually.

**Add an early stopping condition.** If the best score doesn't improve between rounds, stop. My run would have saved a full round's worth of API calls with no accuracy loss.

---

## Closing Thought

The thing that stuck with me most from this experiment isn't the accuracy numbers — it's the *gradient strings*. The model, shown three misclassified tickets, produced natural-language diagnoses that were precise enough to drive genuine prompt improvements. It named the linguistic boundary ("suggestions are opinions, not factual queries"), the structural ambiguity ("escalation intent takes precedence over technical content"), and the domain rule ("billing team owns payment system bugs").

That's the real promise of APO: not just better prompts, but automated documentation of *why* a classifier behaves the way it does. Each gradient is a legible, auditable explanation of a failure mode — the kind of thing that normally lives only in the head of the engineer who spent three weeks debugging production misroutes.

---

*Code and full output logs: [github.com/subhasmitasahoo/auto-prompt-optimization](https://github.com/subhasmitasahoo/auto-prompt-optimization)*  
*Paper: Pryzant et al. (2023), "Automatic Prompt Optimization with Gradient Descent and Beam Search" — [arxiv 2305.03495](https://arxiv.org/pdf/2305.03495)*
