"""
Simulation runner for the APO Customer Support Classifier.

Replaces the Anthropic API with a deterministic mock that mirrors realistic
model behaviour — classification accuracy starts imperfect and improves as
the prompt is refined across rounds.

Run:
    python simulate_run.py

Output is tee'd to both stdout and simulation_output.txt.
"""

import re
import sys
import random
import textwrap
from io import StringIO
from typing import List, Tuple, Dict

# ── patch sys.path so we can import the module files ─────────────────────────
import importlib, types

# We stub out 'anthropic' entirely before importing optimizer.py
mock_anthropic = types.ModuleType("anthropic")

class _FakeMsg:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(text=text)]

class _FakeMessages:
    def create(self, model, max_tokens, messages):
        prompt_text = messages[-1]["content"]
        return _FakeMsg(_mock_llm_call(model, prompt_text, max_tokens))

class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()

mock_anthropic.Anthropic = _FakeClient
sys.modules["anthropic"] = mock_anthropic

# Now safe to import our module
import data as data_mod
import prompts as prompts_mod

CLASSES = data_mod.CLASSES
TRAINING_DATA = data_mod.TRAINING_DATA
EVAL_DATA = data_mod.EVAL_DATA

# ─────────────────────────────────────────────────────────────────────────────
# Realistic mock LLM
# ─────────────────────────────────────────────────────────────────────────────

# Base confusion matrix — (true_label → list of (wrong_label, weight))
# Models which tickets the initial prompt tends to misclassify
_CONFUSION = {
    "Technical Support":      [("Complaint and Escalations", 0.15), ("Billing", 0.05)],
    "Billing":                [("Technical Support", 0.10), ("Complaint and Escalations", 0.08)],
    "General Information":    [("Feedback and Suggestions", 0.12), ("Technical Support", 0.05)],
    "Complaint and Escalations": [("Technical Support", 0.15), ("Feedback and Suggestions", 0.05)],
    "Feedback and Suggestions":  [("General Information", 0.13), ("Technical Support", 0.04)],
}

# After each round of APO the error rates drop — simulate improvement
_ROUND_ERROR_DISCOUNT = {0: 1.0, 1: 0.60, 2: 0.35, 3: 0.18}

_current_round = 0   # bumped by the test harness below
_rng = random.Random(42)


def _classify_ticket(text: str) -> str:
    """Simulate model classification with realistic errors."""
    true_label = None
    for ex in TRAINING_DATA + EVAL_DATA:
        if ex.text == text:
            true_label = ex.label
            break
    if true_label is None:
        return CLASSES[0]

    discount = _ROUND_ERROR_DISCOUNT.get(_current_round, 0.18)
    confusions = _CONFUSION.get(true_label, [])
    for wrong_label, base_prob in confusions:
        if _rng.random() < base_prob * discount:
            return wrong_label
    return true_label


# Pre-crafted gradient feedback strings — realistic and varied
_GRADIENT_BANK = [
    (
        "it does not distinguish between tickets that contain both technical symptoms "
        "AND strong frustration language — those are Complaint and Escalations, not "
        "Technical Support, because the primary intent is escalation not debugging."
    ),
    (
        "it confuses General Information queries with Feedback and Suggestions when "
        "users phrase questions as implicit suggestions (e.g. 'Can you add X?'). "
        "Feedback implies the user is offering an opinion or improvement idea, while "
        "General Information is a neutral factual query."
    ),
    (
        "it lacks class-level signal about billing disputes that mention system errors — "
        "if the root cause is a payment system bug the ticket is still Billing, "
        "not Technical Support, because the resolution path is the billing team."
    ),
    (
        "it treats all polite, positive messages as Feedback and Suggestions, but "
        "messages asking about features or plans are General Information requests "
        "even if they're phrased positively."
    ),
    (
        "it misses the escalation intent in messages that combine technical detail "
        "with demands for management contact, SLA breach mentions, or legal threats — "
        "those are unambiguous Complaint and Escalations regardless of technical content."
    ),
    (
        "it does not handle complex enterprise tickets where multiple issues co-occur. "
        "The primary classification should follow the ticket's opening demand/complaint, "
        "not the longest technical paragraph."
    ),
]

# Pre-crafted improved prompt versions returned by the edit step
_IMPROVED_PROMPTS = [
    textwrap.dedent("""\
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
        Respond with ONLY the class name, nothing else."""),

    textwrap.dedent("""\
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
        Respond with ONLY the class name, nothing else."""),

    textwrap.dedent("""\
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

        Respond with ONLY the class name."""),

    textwrap.dedent("""\
        # Task
        Route the following customer support ticket to the correct team.

        # Teams and what they handle
        | Team | Handle when... |
        |------|---------------|
        | Technical Support | System errors, bugs, crashes, authentication failures, slow performance, broken integrations |
        | Billing | Payments, invoices, refunds, pricing, subscription changes, billing system errors |
        | General Information | Factual questions about product features, policies, hours, plans |
        | Complaint and Escalations | Anger, demands for management, legal threats, SLA violations, repeated unresolved issues |
        | Feedback and Suggestions | Feature ideas, praise, UI feedback, improvement requests |

        # Key distinctions
        - A billing system bug → **Billing** (not Technical Support)
        - A ticket with both technical issues AND escalation language → **Complaint and Escalations**
        - "Can you add X feature?" → **Feedback and Suggestions** (not General Information)
        - "How does X work?" → **General Information** (not Feedback and Suggestions)

        Respond with ONLY the team name (class name) from the table above."""),
]
_prompt_iter = iter(_IMPROVED_PROMPTS)


def _mock_llm_call(model: str, prompt_text: str, max_tokens: int) -> str:
    """Route to the right mock response based on call type."""
    # Gradient call: return reason feedback
    if "Give" in prompt_text and "reasons why the prompt" in prompt_text:
        reasons = _rng.sample(_GRADIENT_BANK, k=min(3, len(_GRADIENT_BANK)))
        return "\n\n".join(f"<START>{r}<END>" for r in reasons)

    # Edit call: return new prompt candidates
    if "write" in prompt_text.lower() and "improved prompts" in prompt_text.lower():
        candidates = []
        for _ in range(2):
            try:
                candidates.append(next(_prompt_iter))
            except StopIteration:
                candidates.append(_IMPROVED_PROMPTS[-1])
        return "\n\n".join(f"<START>\n{p}\n<END>" for p in candidates)

    # Classification call
    ticket_match = re.search(r"Ticket:\s*(.+)$", prompt_text, re.DOTALL)
    if ticket_match:
        ticket_text = ticket_match.group(1).strip()
        return _classify_ticket(ticket_text)

    return CLASSES[0]


# ─────────────────────────────────────────────────────────────────────────────
# Import optimizer AFTER mock is in place
# ─────────────────────────────────────────────────────────────────────────────
import optimizer as opt_mod   # noqa: E402  (intentional late import)


# Monkey-patch _current_round into optimizer's classify so error rate tracks round
_original_classify = opt_mod.classify

def _round_aware_classify(prompt, ticket):
    return _original_classify(prompt, ticket)

opt_mod.classify = _round_aware_classify


# ─────────────────────────────────────────────────────────────────────────────
# Tee output to both stdout and a file
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def run_simulation():
    global _current_round

    output_path = "simulation_output.txt"
    log_file = open(output_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        # ── Seed evaluation ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  AUTO PROMPT OPTIMIZATION — Customer Support Classifier")
        print("  [SIMULATION MODE — mock LLM, deterministic seed 42]")
        print("=" * 60)
        print(f"\n📌 Seed prompt:\n{prompts_mod.INITIAL_PROMPT}\n")

        train_sample = _rng.sample(TRAINING_DATA, 30)
        acc0, errors0 = opt_mod.evaluate(prompts_mod.INITIAL_PROMPT, train_sample)
        print(f"📊 Seed accuracy on 30-ticket train sample: {acc0:.1%}  ({len(errors0)} errors)\n")

        # Show which tickets the seed prompt misclassifies
        if errors0:
            print("  Seed prompt misclassifications:")
            for e in errors0:
                short = e["text"][:70] + "…" if len(e["text"]) > 70 else e["text"]
                print(f"    ✗  [{e['expected']}] → [{e['predicted']}]  \"{short}\"")

        # ── APO rounds ───────────────────────────────────────────────────────
        beam: List[Tuple[float, str]] = [(acc0, prompts_mod.INITIAL_PROMPT)]
        history = []

        for round_num in range(1, 4):
            _current_round = round_num
            print(f"\n{'─'*60}")
            print(f"  Round {round_num} / 3")
            print(f"{'─'*60}")

            candidates = list(beam)

            for rank, (beam_acc, beam_prompt) in enumerate(beam):
                print(f"\n  ▸ Beam prompt #{rank+1}  (train acc = {beam_acc:.1%})")
                _, errors = opt_mod.evaluate(beam_prompt, train_sample)
                if not errors:
                    print("    ✅ No errors — skipping gradient step.")
                    continue

                print(f"    🔍 Requesting 3 gradient feedbacks…")
                gradients = opt_mod.get_gradients(beam_prompt, errors, num_feedbacks=3)

                # ── Print every gradient string ──────────────────────────────
                for g_idx, grad in enumerate(gradients):
                    print(f"\n    ┌─ Gradient {g_idx+1} ─────────────────────────────────")
                    for line in grad.splitlines():
                        print(f"    │  {line}")
                    print(f"    └────────────────────────────────────────────────────")

                print(f"\n    ✏️  Generating {len(gradients) * 2} candidate prompt(s) (2 per gradient)…")

                for g_idx, grad in enumerate(gradients):
                    new_prompts = opt_mod.generate_new_prompts(beam_prompt, errors, grad, 2)
                    for p_idx, np_ in enumerate(new_prompts):
                        cand_acc, _ = opt_mod.evaluate(np_, train_sample)
                        candidates.append((cand_acc, np_))
                        flag = " ⭐" if cand_acc == 1.0 else (" ⚠️" if cand_acc < acc0 else "")
                        print(f"\n      ── Gradient {g_idx+1} · Candidate {p_idx+1}  (train acc = {cand_acc:.1%}{flag}) ──")
                        for line in np_.splitlines():
                            print(f"         {line}")
                        print()

            # Keep top-3 unique
            seen, unique = set(), []
            for score, p in sorted(candidates, reverse=True):
                if p not in seen:
                    seen.add(p)
                    unique.append((score, p))
            beam = unique[:3]

            best_acc, best_prompt = beam[0]
            print(f"\n  🏆 Best prompt this round — train acc: {best_acc:.1%}")
            print(f"  Preview: {best_prompt.splitlines()[0][:80]}…")
            history.append({"round": round_num, "best_acc": best_acc})

        # ── Final eval on held-out set ───────────────────────────────────────
        _current_round = 3
        final_prompt    = beam[0][1]
        final_train_acc = beam[0][0]
        eval_acc, eval_errors = opt_mod.evaluate(final_prompt, EVAL_DATA)

        print("\n" + "=" * 60)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"\n🎯 Final optimized prompt:\n")
        print(final_prompt)
        print(f"\n{'─'*60}")
        print(f"📊 Final train accuracy : {final_train_acc:.1%}")
        print(f"📊 Final eval  accuracy : {eval_acc:.1%}  ({len(eval_errors)} errors on 15 held-out tickets)")

        if eval_errors:
            print(f"\n  Remaining eval errors:")
            for e in eval_errors:
                short = e["text"][:70] + "…" if len(e["text"]) > 70 else e["text"]
                print(f"    ✗  [{e['expected']}] → [{e['predicted']}]  \"{short}\"")
        else:
            print("\n  🎉 Zero errors on the held-out eval set!")

        print(f"\n{'─'*60}")
        print("  Accuracy progression:")
        print(f"    Seed  : {acc0:.1%}")
        for h in history:
            print(f"    Round {h['round']} : {h['best_acc']:.1%}")
        print(f"    Eval  : {eval_acc:.1%}  (held-out)")
        print(f"{'─'*60}\n")

    finally:
        sys.stdout = sys.__stdout__
        log_file.close()

    print(f"\n✅ Full output saved to {output_path}")
    return {
        "seed_accuracy":  acc0,
        "history":        history,
        "final_prompt":   final_prompt,
        "train_accuracy": final_train_acc,
        "eval_accuracy":  eval_acc,
    }


if __name__ == "__main__":
    result = run_simulation()
