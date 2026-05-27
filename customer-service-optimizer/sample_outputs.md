# Sample Outputs — APO Customer Support Classifier

> **Run config**: 3 rounds · beam size 3 · 3 gradient feedbacks/prompt · 2 candidates/gradient  
> **Data**: 30-ticket training sample · 15-ticket held-out eval set  
> **Mode**: Simulation (`python simulate_run.py`, deterministic seed 42)

---

## Accuracy Summary

| Stage | Train acc | Errors | Notes |
|---|---|---|---|
| **Seed** | 90.0% | 3 | All errors on General Info ↔ Feedback boundary |
| **Round 1** | 96.7% | 1 | Gradient fixed intent distinction |
| **Round 2** | 100.0% | 0 | Convergence — first perfect train score |
| **Round 3** | 100.0% | 0 | No candidate beat current best; maintained |
| **Eval (held-out)** | **100.0%** | **0** | Zero errors on 15 unseen tickets |

---

## Seed Prompt

```
# Task
Categorize the customer support ticket.

# Output format
Classify into one of these classes: 'Technical Support', 'Billing', 'General Information',
'Complaint and Escalations', 'Feedback and Suggestions'.
Respond with ONLY the class name, nothing else.
```

**Seed misclassifications (3 errors):**

| # | True label | Predicted | Ticket (truncated) |
|---|---|---|---|
| 1 | General Information | Feedback and Suggestions | "Can you tell me more about your enterprise plan features?" |
| 2 | Feedback and Suggestions | General Information | "Your onboarding process was smooth, but a video tutorial would be helpful…" |
| 3 | Feedback and Suggestions | General Information | "Consider adding bulk upload functionality - it would save us a lot of time…" |

---

## Round 1

### Gradient Feedback (on Beam #1, train acc = 90.0%)

| # | Gradient feedback |
|---|---|
| G1 | it treats all polite, positive messages as Feedback and Suggestions, but messages asking about features or plans are General Information requests even if they're phrased positively. |
| G2 | it does not distinguish between tickets that contain both technical symptoms AND strong frustration language — those are Complaint and Escalations, not Technical Support, because the primary intent is escalation not debugging. |
| G3 | it confuses General Information queries with Feedback and Suggestions when users phrase questions as implicit suggestions (e.g. 'Can you add X?'). Feedback implies the user is offering an opinion or improvement idea, while General Information is a neutral factual query. |

### Edit Candidates

| Candidate | From gradient | Train acc | Result |
|---|---|---|---|
| R1·G1·C1 | G1 | 96.7% | ✅ Promoted to beam |
| R1·G1·C2 | G1 | 96.7% | ✅ Promoted to beam |
| R1·G2·C1 | G2 | 90.0% | — Tied with seed; discarded |
| R1·G2·C2 | G2 | 90.0% | — Tied with seed; discarded |
| R1·G3·C1 | G3 | 86.7% | ⚠️ Regressed below seed; discarded |
| R1·G3·C2 | G3 | 96.7% | ✅ Promoted to beam |

<details>
<summary><strong>R1·G1·C1</strong> — train acc 96.7% (click to expand)</summary>

```
# Task
Categorize the customer support ticket into exactly one class.

# Class definitions
- **Technical Support**: The customer reports a software bug, error message,
  integration failure, performance problem, or configuration issue that requires
  the engineering or technical team to investigate and fix.
- **Billing**: Anything related to charges, invoices, refunds, payment methods,
  subscription changes, pricing, or financial discrepancies — even if a payment
  system has a technical glitch.
- **General Information**: A neutral factual question about how the product works,
  what features exist, what plans are available, or what the company's policies are.
  The customer is seeking information, not reporting a problem or offering an opinion.
- **Complaint and Escalations**: The customer expresses strong dissatisfaction,
  demands management contact, mentions legal action, references repeated failures,
  or threatens to cancel/leave. These often contain technical or billing details
  but the primary intent is escalation.
- **Feedback and Suggestions**: The customer shares an opinion, idea, or
  improvement request. This includes both praise and constructive suggestions.

# Output format
Respond with ONLY the class name, nothing else.
```

</details>

<details>
<summary><strong>R1·G1·C2</strong> — train acc 96.7% (click to expand)</summary>

```
# Task
You are a customer support ticket router. Assign each ticket to the single
most appropriate class below.

# Routing rules
1. If the ticket reports a broken feature, error, or technical malfunction → **Technical Support**
2. If the ticket is about money: charges, invoices, refunds, payment, pricing → **Billing**
   (even if a billing system has a bug, route to Billing not Technical Support)
3. If the ticket asks a neutral question about the product, features, or policies → **General Information**
4. If the ticket contains threats, demands for management, legal language, or expresses
   severe dissatisfaction after repeated failures → **Complaint and Escalations**
   (prioritise this over Technical Support if BOTH apply)
5. If the ticket praises the product, suggests an improvement, or requests a new feature → **Feedback and Suggestions**
   (distinguish from General Information: suggestions are opinions, not factual queries)

# Output format
Respond with ONLY the class name, nothing else.
```

</details>

<details>
<summary><strong>R1·G2·C1</strong> — train acc 90.0% · <strong>R1·G2·C2</strong> — train acc 90.0% (click to expand)</summary>

Both generated from gradient G2 (escalation intent). Both scored the same as the seed — the escalation distinction was already correct in the seed sample; gradient G2 addressed a problem that wasn't present in the 30-ticket sample, so the candidates brought no improvement.

```
# Task
Classify the customer support ticket into one of five categories.

# Categories
**Technical Support** — Bug reports, crashes, login failures, sync errors,
slow performance, integration problems, API errors. The user needs a technical fix.

**Billing** — Duplicate charges, refund requests, invoice questions, payment
method updates, subscription changes, pricing disputes, missing discounts.
Route here even when the payment processing itself has a technical error.

**General Information** — Questions about business hours, shipping times,
plan features, file format support, compliance certifications. Neutral
information-seeking; no problem to fix and no opinion offered.

**Complaint and Escalations** — Tickets with escalation signals: demands to
speak to a manager, mentions of legal action or consumer protection bodies,
SLA breach references, expressions of intent to cancel, or documentation of
financial losses. These may also describe technical problems but escalation
intent takes precedence.

**Feedback and Suggestions** — Feature requests, UI improvement ideas,
compliments, bug suggestions, praise for specific team members. The customer
is offering an opinion or idea, not requesting resolution of a problem.

# Decision tie-breaker
When a ticket could fit two classes, choose the one whose team would own
the next action (e.g. a frustrated billing dispute → Billing, a repeated
escalating technical failure → Complaint and Escalations).

Respond with ONLY the class name.
```

</details>

<details>
<summary><strong>R1·G3·C1</strong> — train acc 86.7% ⚠️ (regressed — click to expand)</summary>

Generated from gradient G3, which focused on the Feedback vs General Information boundary — but the prompt produced is identical in structure to the G2 candidates (the mock returns prompts sequentially). Scored **below the seed** (86.7% vs 90.0%) — beam search discarded it immediately.

```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                               |
|---------------------------|------------------------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures, slow performance,     |
|                           | broken integrations                                                          |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes, billing system   |
|                           | errors                                                                       |
| General Information       | Factual questions about product features, policies, hours, plans             |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations, repeated       |
|                           | unresolved issues                                                            |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests                     |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

</details>

---

## Round 2

Beam entering Round 2: **3 prompts all at 96.7%**

### Gradient Feedback Summary

| Beam | # | Gradient feedback |
|---|---|---|
| #1 | G1 | it does not handle complex enterprise tickets where multiple issues co-occur. The primary classification should follow the ticket's opening demand/complaint, not the longest technical paragraph. |
| #1 | G2 | it confuses General Information queries with Feedback and Suggestions when users phrase questions as implicit suggestions (e.g. 'Can you add X?'). Feedback implies the user is offering an opinion or improvement idea, while General Information is a neutral factual query. |
| #1 | G3 | it does not distinguish between tickets that contain both technical symptoms AND strong frustration language — those are Complaint and Escalations, not Technical Support, because the primary intent is escalation not debugging. |
| #2 | G1 | it treats all polite, positive messages as Feedback and Suggestions, but messages asking about features or plans are General Information requests even if they're phrased positively. |
| #2 | G2 | it lacks class-level signal about billing disputes that mention system errors — if the root cause is a payment system bug the ticket is still Billing, not Technical Support, because the resolution path is the billing team. |
| #2 | G3 | it does not distinguish between tickets that contain both technical symptoms AND strong frustration language — those are Complaint and Escalations, not Technical Support, because the primary intent is escalation not debugging. |
| #3 | — | ✅ No errors on this beam prompt — gradient step skipped |

### Edit Candidates

| Candidate | From | Train acc | Result |
|---|---|---|---|
| R2·B1·G1·C1 | Beam 1, G1 | 93.3% | — |
| R2·B1·G1·C2 | Beam 1, G1 | 96.7% | — |
| R2·B1·G2·C1 | Beam 1, G2 | 93.3% | — |
| R2·B1·G2·C2 | Beam 1, G2 | 90.0% | — |
| R2·B1·G3·C1 | Beam 1, G3 | 96.7% | — |
| **R2·B1·G3·C2** | **Beam 1, G3** | **100.0%** | ⭐ **First 100% — promoted** |
| R2·B2·G1·C1 | Beam 2, G1 | 93.3% | — |
| R2·B2·G1·C2 | Beam 2, G1 | 93.3% | — |
| **R2·B2·G2·C1** | **Beam 2, G2** | **100.0%** | ⭐ **Promoted** |
| **R2·B2·G2·C2** | **Beam 2, G2** | **100.0%** | ⭐ **Promoted** |
| R2·B2·G3·C1 | Beam 2, G3 | 93.3% | — |
| R2·B2·G3·C2 | Beam 2, G3 | 93.3% | — |

<details>
<summary><strong>R2·B1·G3·C2</strong> — first candidate to hit 100% ⭐ (click to expand)</summary>

Generated from Beam #1's Gradient 3: *"escalation intent in tickets that combine technical detail with management demands."*

```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                               |
|---------------------------|------------------------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures, slow performance,     |
|                           | broken integrations                                                          |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes, billing system   |
|                           | errors                                                                       |
| General Information       | Factual questions about product features, policies, hours, plans             |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations, repeated       |
|                           | unresolved issues                                                            |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests                     |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

</details>

---

## Round 3

Beam entering Round 3: **1 prompt at 100%, 2 prompts at 96.7%**

### Gradient Feedback Summary

| Beam | # | Gradient feedback |
|---|---|---|
| #1 | G1 | it lacks class-level signal about billing disputes that mention system errors — if the root cause is a payment system bug the ticket is still Billing, not Technical Support, because the resolution path is the billing team. |
| #1 | G2 | it misses the escalation intent in messages that combine technical detail with demands for management contact, SLA breach mentions, or legal threats — those are unambiguous Complaint and Escalations regardless of technical content. |
| #1 | G3 | it does not handle complex enterprise tickets where multiple issues co-occur. The primary classification should follow the ticket's opening demand/complaint, not the longest technical paragraph. |
| #2 | G1 | it misses the escalation intent in messages that combine technical detail with demands for management contact, SLA breach mentions, or legal threats — those are unambiguous Complaint and Escalations regardless of technical content. |
| #2 | G2 | it confuses General Information queries with Feedback and Suggestions when users phrase questions as implicit suggestions. |
| #2 | G3 | it does not handle complex enterprise tickets where multiple issues co-occur. |
| #3 | G1 | it treats all polite, positive messages as Feedback and Suggestions, but messages asking about features or plans are General Information requests. |
| #3 | G2 | it lacks class-level signal about billing disputes that mention system errors. |
| #3 | G3 | it misses the escalation intent in messages that combine technical detail with management demands or legal threats. |

### Edit Candidates

| Candidate | Train acc | Result |
|---|---|---|
| R3·B1·G1·C1 | 100.0% | ⭐ |
| R3·B1·G1·C2 | 100.0% | ⭐ |
| R3·B1·G2·C1 | 100.0% | ⭐ |
| R3·B1·G2·C2 | 96.7% | — |
| R3·B1·G3·C1 | 100.0% | ⭐ |
| R3·B1·G3·C2 | 96.7% | — |
| R3·B2·G1·C1 | 96.7% | — |
| R3·B2·G1·C2 | 96.7% | — |
| R3·B2·G2·C1 | 100.0% | ⭐ |
| R3·B2·G2·C2 | 100.0% | ⭐ |
| R3·B2·G3·C1 | 93.3% | — |
| R3·B2·G3·C2 | 100.0% | ⭐ |
| R3·B3·G1·C1 | 100.0% | ⭐ |
| R3·B3·G1·C2 | 96.7% | — |
| R3·B3·G2·C1 | 100.0% | ⭐ |
| R3·B3·G2·C2 | 100.0% | ⭐ |
| R3·B3·G3·C1 | 96.7% | — |
| R3·B3·G3·C2 | 100.0% | ⭐ |

**Round 3 observation**: 11 of 18 candidates scored 100%. No candidate beat the existing best — beam retained the Round 2 winner. This is the early-stopping signal: if the best score doesn't improve, terminate the loop.

---

## Final Optimized Prompt

```
# Task
Route the following customer support ticket to the correct team.

# Teams and what they handle
| Team                      | Handle when...                                                               |
|---------------------------|------------------------------------------------------------------------------|
| Technical Support         | System errors, bugs, crashes, authentication failures, slow performance,     |
|                           | broken integrations                                                          |
| Billing                   | Payments, invoices, refunds, pricing, subscription changes, billing system   |
|                           | errors                                                                       |
| General Information       | Factual questions about product features, policies, hours, plans             |
| Complaint and Escalations | Anger, demands for management, legal threats, SLA violations, repeated       |
|                           | unresolved issues                                                            |
| Feedback and Suggestions  | Feature ideas, praise, UI feedback, improvement requests                     |

# Key distinctions
- A billing system bug → **Billing** (not Technical Support)
- A ticket with both technical issues AND escalation language → **Complaint and Escalations**
- "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
- "How does X work?" → **General Information** (not Feedback and Suggestions)

Respond with ONLY the team name (class name) from the table above.
```

**Final scores:**
- Train accuracy: **100.0%**
- Eval accuracy: **100.0%** (0 errors on 15 held-out tickets)

---

## All Gradient Strings (Deduplicated)

These are the unique gradient reasons the optimizer produced across all 3 rounds:

| ID | Gradient text |
|---|---|
| **g-a** | it treats all polite, positive messages as Feedback and Suggestions, but messages asking about features or plans are General Information requests even if they're phrased positively. |
| **g-b** | it does not distinguish between tickets that contain both technical symptoms AND strong frustration language — those are Complaint and Escalations, not Technical Support, because the primary intent is escalation not debugging. |
| **g-c** | it confuses General Information queries with Feedback and Suggestions when users phrase questions as implicit suggestions (e.g. 'Can you add X?'). Feedback implies the user is offering an opinion or improvement idea, while General Information is a neutral factual query. |
| **g-d** | it lacks class-level signal about billing disputes that mention system errors — if the root cause is a payment system bug the ticket is still Billing, not Technical Support, because the resolution path is the billing team. |
| **g-e** | it misses the escalation intent in messages that combine technical detail with demands for management contact, SLA breach mentions, or legal threats — those are unambiguous Complaint and Escalations regardless of technical content. |
| **g-f** | it does not handle complex enterprise tickets where multiple issues co-occur. The primary classification should follow the ticket's opening demand/complaint, not the longest technical paragraph. |

**Which gradients actually drove the winning prompt:**  
`g-a` / `g-c` → fixed General Info vs Feedback boundary (the seed's only errors)  
`g-b` / `g-e` → added escalation-wins-over-technical rule  
`g-d` → added billing-system-bug disambiguation rule  
`g-f` → added "opening demand" tie-breaker for composite tickets
