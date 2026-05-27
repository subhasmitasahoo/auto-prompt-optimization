# Sample Outputs — APO Customer Support Classifier

> **Run config**: 3 rounds · beam size 3 · 3 gradient feedbacks/prompt · 2 candidates/gradient  
> **Data**: 30-ticket training sample · 15-ticket held-out eval set  
> **Mode**: Simulation (deterministic mock LLM, seed 42)  
> **Runner**: `python simulate_run.py`

---

## Full Console Output

```
============================================================
  AUTO PROMPT OPTIMIZATION — Customer Support Classifier
  [SIMULATION MODE — mock LLM, deterministic seed 42]
============================================================

📌 Seed prompt:
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing', 'General Information',
'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.

📊 Seed accuracy on 30-ticket train sample: 90.0%  (3 errors)

  Seed prompt misclassifications:
    ✗  [General Information] → [Feedback and Suggestions]
       "Can you tell me more about your enterprise plan features?"
    ✗  [Feedback and Suggestions] → [General Information]
       "Your onboarding process was smooth, but a video tutorial would be helpful…"
    ✗  [Feedback and Suggestions] → [General Information]
       "Consider adding bulk upload functionality - it would save us a lot of time…"

────────────────────────────────────────────────────────────
  Round 1 / 3
────────────────────────────────────────────────────────────

  ▸ Beam prompt #1  (train acc = 90.0%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 96.7%
      Gradient 1, candidate 2: train acc = 96.7%
      Gradient 2, candidate 1: train acc = 90.0%
      Gradient 2, candidate 2: train acc = 90.0%
      Gradient 3, candidate 1: train acc = 86.7%   ← regression (discarded by beam)
      Gradient 3, candidate 2: train acc = 96.7%

  🏆 Best prompt this round — train acc: 96.7%

────────────────────────────────────────────────────────────
  Round 2 / 3
────────────────────────────────────────────────────────────

  ▸ Beam prompt #1  (train acc = 96.7%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 93.3%
      Gradient 1, candidate 2: train acc = 96.7%
      Gradient 2, candidate 1: train acc = 93.3%
      Gradient 2, candidate 2: train acc = 90.0%
      Gradient 3, candidate 1: train acc = 96.7%
      Gradient 3, candidate 2: train acc = 100.0%  ← new best

  ▸ Beam prompt #2  (train acc = 96.7%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 93.3%
      Gradient 1, candidate 2: train acc = 93.3%
      Gradient 2, candidate 1: train acc = 100.0%
      Gradient 2, candidate 2: train acc = 100.0%
      Gradient 3, candidate 1: train acc = 93.3%
      Gradient 3, candidate 2: train acc = 93.3%

  ▸ Beam prompt #3  (train acc = 96.7%)
    ✅ No errors — skipping gradient step.

  🏆 Best prompt this round — train acc: 100.0%

────────────────────────────────────────────────────────────
  Round 3 / 3
────────────────────────────────────────────────────────────

  ▸ Beam prompt #1  (train acc = 100.0%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 100.0%
      Gradient 1, candidate 2: train acc = 100.0%
      Gradient 2, candidate 1: train acc = 100.0%
      Gradient 2, candidate 2: train acc = 96.7%
      Gradient 3, candidate 1: train acc = 100.0%
      Gradient 3, candidate 2: train acc = 96.7%

  ▸ Beam prompt #2  (train acc = 96.7%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 96.7%
      Gradient 1, candidate 2: train acc = 96.7%
      Gradient 2, candidate 1: train acc = 100.0%
      Gradient 2, candidate 2: train acc = 100.0%
      Gradient 3, candidate 1: train acc = 93.3%
      Gradient 3, candidate 2: train acc = 100.0%

  ▸ Beam prompt #3  (train acc = 96.7%)
    🔍 Requesting 3 gradient feedbacks…
    ✏️  Generating 6 candidate prompt(s) (2 per gradient)…
      Gradient 1, candidate 1: train acc = 100.0%
      Gradient 1, candidate 2: train acc = 96.7%
      Gradient 2, candidate 1: train acc = 100.0%
      Gradient 2, candidate 2: train acc = 100.0%
      Gradient 3, candidate 1: train acc = 96.7%
      Gradient 3, candidate 2: train acc = 100.0%

  🏆 Best prompt this round — train acc: 100.0%

============================================================
  OPTIMIZATION COMPLETE
============================================================

🎯 Final optimized prompt:

# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                        |
|---------------------------|-----------------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures,                |
|                           | slow performance, broken integrations                                 |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes,           |
|                           | billing system errors                                                 |
| General Information       | Factual questions about product features, policies, hours, plans      |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations,         |
|                           | repeated unresolved issues                                            |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests              |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.

────────────────────────────────────────────────────────────
📊 Final train accuracy : 100.0%
📊 Final eval  accuracy : 100.0%  (0 errors on 15 held-out tickets)

  🎉 Zero errors on the held-out eval set!

────────────────────────────────────────────────────────────
  Accuracy progression:
    Seed    : 90.0%
    Round 1 : 96.7%
    Round 2 : 100.0%
    Round 3 : 100.0%
    Eval    : 100.0%  (held-out)
────────────────────────────────────────────────────────────
```

---

## Accuracy Progression

| Stage    | Train acc (30-sample) | Notes                                      |
|----------|-----------------------|--------------------------------------------|
| Seed     | 90.0% (3 errors)      | All errors on General Info vs Feedback     |
| Round 1  | 96.7% (1 error)       | Gradient fixed the Info/Feedback boundary  |
| Round 2  | 100.0% (0 errors)     | Convergence — beam reached perfect train   |
| Round 3  | 100.0% (0 errors)     | Maintained; no candidate beat current best |
| **Eval** | **100.0% (0 errors)** | Zero errors on the 15 held-out tickets     |

---

## Seed Prompt vs Final Prompt

### Seed
```
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing',
'General Information', 'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.
```

### Final (after 2 rounds)
```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                  |
|---------------------------|-----------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures,          |
|                           | slow performance, broken integrations                           |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes,     |
|                           | billing system errors                                           |
| General Information       | Factual questions about product features, policies, hours, plans|
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations,   |
|                           | repeated unresolved issues                                      |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests        |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

---

## Candidate Scores Across All Rounds

| Round | Beam # | Gradient | Candidate | Train acc |
|-------|--------|----------|-----------|-----------|
| 1     | 1      | G1       | C1        | 96.7%     |
| 1     | 1      | G1       | C2        | 96.7%     |
| 1     | 1      | G2       | C1        | 90.0%     |
| 1     | 1      | G2       | C2        | 90.0%     |
| 1     | 1      | G3       | C1        | 86.7% ⚠️  |
| 1     | 1      | G3       | C2        | 96.7%     |
| 2     | 1      | G1       | C1        | 93.3%     |
| 2     | 1      | G1       | C2        | 96.7%     |
| 2     | 1      | G2       | C1        | 93.3%     |
| 2     | 1      | G2       | C2        | 90.0%     |
| 2     | 1      | G3       | C1        | 96.7%     |
| 2     | 1      | G3       | C2        | **100.0%** ⭐ |
| 2     | 2      | G1       | C1        | 93.3%     |
| 2     | 2      | G1       | C2        | 93.3%     |
| 2     | 2      | G2       | C1        | **100.0%** ⭐ |
| 2     | 2      | G2       | C2        | **100.0%** ⭐ |
| 2     | 2      | G3       | C1        | 93.3%     |
| 2     | 2      | G3       | C2        | 93.3%     |
| 3     | 1–3    | G1–G3    | C1–C2     | 93.3–100% |

⚠️ = regressed below seed · ⭐ = first candidate to reach 100%
