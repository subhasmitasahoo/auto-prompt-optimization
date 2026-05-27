"""
Core Auto Prompt Optimization (APO) engine.

Algorithm (from https://arxiv.org/pdf/2305.03495):
  1. Evaluate current prompt(s) on training data → collect errors
  2. Run gradient prompt  → get N reasons why the prompt fails
  3. Run edit prompt      → generate M improved prompt candidates per reason
  4. Evaluate all candidates, keep the top-K (beam)
  5. Repeat for num_rounds
"""

import re
import random
import anthropic

from typing import List, Tuple, Dict
from data import Example, CLASSES
from prompts import GRADIENT_PROMPT_TEMPLATE, EDIT_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# Models — fast model for bulk classification, capable model for meta-prompting
# ---------------------------------------------------------------------------
CLASSIFY_MODEL  = "claude-haiku-4-5-20251001"   # cheap, fast — called many times
OPTIMIZE_MODEL  = "claude-sonnet-4-6"            # smarter   — gradient + edit steps

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def parse_prediction(response: str, classes: List[str] = CLASSES) -> str:
    """
    Extract a class label from a raw model response.

    Tries (in order):
      1. Exact case-insensitive match
      2. Any class name appearing as a substring in the response
      3. Returns the stripped response unchanged (will count as wrong)
    """
    cleaned = response.strip()
    lower   = cleaned.lower()

    for cls in classes:
        if cls.lower() == lower:
            return cls

    for cls in classes:
        if cls.lower() in lower:
            return cls

    return cleaned


def classify(prompt: str, ticket: str) -> str:
    """Return the predicted class for a single support ticket."""
    client = _get_client()
    message = client.messages.create(
        model=CLASSIFY_MODEL,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\nTicket: {ticket}",
            }
        ],
    )
    raw = message.content[0].text
    return parse_prediction(raw)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    prompt: str,
    data: List[Example],
) -> Tuple[float, List[Dict]]:
    """
    Evaluate a prompt on a dataset.

    Returns:
        accuracy : float in [0, 1]
        errors   : list of dicts with keys text / expected / predicted
    """
    correct = 0
    errors: List[Dict] = []

    for example in data:
        predicted = classify(prompt, example.text)
        if predicted.lower() == example.label.lower():
            correct += 1
        else:
            errors.append(
                {
                    "text":      example.text,
                    "expected":  example.label,
                    "predicted": predicted,
                }
            )

    accuracy = correct / len(data) if data else 0.0
    return accuracy, errors


# ---------------------------------------------------------------------------
# Gradient step
# ---------------------------------------------------------------------------

def format_errors(errors: List[Dict], max_errors: int = 5) -> str:
    """Render a capped list of errors as a numbered string."""
    sample = errors[:max_errors]
    lines  = []
    for i, err in enumerate(sample, 1):
        short_text = err["text"][:200] + "…" if len(err["text"]) > 200 else err["text"]
        lines.append(
            f'{i}. Text: "{short_text}"\n'
            f'   Expected : {err["expected"]}\n'
            f'   Predicted: {err["predicted"]}'
        )
    return "\n\n".join(lines)


def get_gradients(
    prompt: str,
    errors: List[Dict],
    num_feedbacks: int = 3,
) -> List[str]:
    """
    Ask the optimizer model why the current prompt gets examples wrong.

    Returns a list of gradient strings (reasons), extracted from
    <START>…<END> blocks in the model response.
    """
    if not errors:
        return []

    client         = _get_client()
    error_string   = format_errors(errors)
    gradient_query = GRADIENT_PROMPT_TEMPLATE.format(
        prompt=prompt,
        error_string=error_string,
        num_feedbacks=num_feedbacks,
    )

    message = client.messages.create(
        model=OPTIMIZE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": gradient_query}],
    )
    response  = message.content[0].text
    gradients = re.findall(r"<START>(.*?)<END>", response, re.DOTALL)
    return [g.strip() for g in gradients]


# ---------------------------------------------------------------------------
# Edit step
# ---------------------------------------------------------------------------

def generate_new_prompts(
    prompt: str,
    errors: List[Dict],
    gradient: str,
    steps_per_gradient: int = 2,
) -> List[str]:
    """
    Given a gradient reason, generate `steps_per_gradient` improved prompts.

    Returns a list of new prompt strings extracted from <START>…<END> blocks.
    """
    client    = _get_client()
    error_str = format_errors(errors)
    edit_query = EDIT_PROMPT_TEMPLATE.format(
        prompt=prompt,
        error_str=error_str,
        gradient=gradient,
        steps_per_gradient=steps_per_gradient,
    )

    message = client.messages.create(
        model=OPTIMIZE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": edit_query}],
    )
    response    = message.content[0].text
    new_prompts = re.findall(r"<START>(.*?)<END>", response, re.DOTALL)
    return [p.strip() for p in new_prompts]


# ---------------------------------------------------------------------------
# Main APO loop
# ---------------------------------------------------------------------------

def run_apo(
    initial_prompt: str,
    train_data: List[Example],
    eval_data: List[Example],
    num_rounds: int        = 3,
    beam_size: int         = 3,
    steps_per_gradient: int = 2,
    num_feedbacks: int     = 3,
    max_train_sample: int  = 30,
    verbose: bool          = True,
) -> Dict:
    """
    Run the Auto Prompt Optimization loop.

    Args:
        initial_prompt     : The starting (seed) prompt.
        train_data         : Labeled examples used during optimization.
        eval_data          : Held-out set for final scoring — never used to guide search.
        num_rounds         : How many optimization rounds to run.
        beam_size          : Number of top prompts kept between rounds.
        steps_per_gradient : New prompts generated per gradient reason.
        num_feedbacks      : Gradient reasons requested per prompt.
        max_train_sample   : Cap on training examples used per evaluation call
                             (speeds things up; set to len(train_data) for full eval).
        verbose            : Print progress to stdout.

    Returns:
        dict with keys:
          best_prompt        : str
          best_train_accuracy: float
          best_eval_accuracy : float
          history            : list of per-round dicts
    """

    def _log(msg: str):
        if verbose:
            print(msg)

    # Subsample training data for speed when the set is large
    train_sample = (
        random.sample(train_data, max_train_sample)
        if len(train_data) > max_train_sample
        else train_data
    )

    # ── Round 0: seed the beam ───────────────────────────────────────────────
    _log("\n" + "=" * 60)
    _log("  AUTO PROMPT OPTIMIZATION — Customer Support Classifier")
    _log("=" * 60)
    _log(f"\n📌 Initial prompt:\n{initial_prompt}\n")

    beam: List[Tuple[float, str]] = []   # (accuracy, prompt)

    acc, errors = evaluate(initial_prompt, train_sample)
    beam.append((acc, initial_prompt))
    _log(f"📊 Seed accuracy on train sample: {acc:.1%}  ({len(errors)} errors)")

    history = []

    # ── Optimization rounds ──────────────────────────────────────────────────
    for round_num in range(1, num_rounds + 1):
        _log(f"\n{'─'*60}")
        _log(f"  Round {round_num} / {num_rounds}")
        _log(f"{'─'*60}")

        candidates: List[Tuple[float, str]] = list(beam)

        for rank, (beam_acc, beam_prompt) in enumerate(beam):
            _log(f"\n  ▸ Beam prompt #{rank+1}  (train acc = {beam_acc:.1%})")

            _, errors = evaluate(beam_prompt, train_sample)
            if not errors:
                _log("    ✅ No errors — skipping gradient step.")
                continue

            _log(f"    🔍 Getting {num_feedbacks} gradient feedback(s)…")
            gradients = get_gradients(beam_prompt, errors, num_feedbacks)
            _log(f"    ✏️  Generating {len(gradients) * steps_per_gradient} candidate prompt(s)…")

            for g_idx, gradient in enumerate(gradients):
                new_prompts = generate_new_prompts(
                    beam_prompt, errors, gradient, steps_per_gradient
                )
                for p_idx, new_prompt in enumerate(new_prompts):
                    cand_acc, _ = evaluate(new_prompt, train_sample)
                    candidates.append((cand_acc, new_prompt))
                    _log(
                        f"      Gradient {g_idx+1}, candidate {p_idx+1}: "
                        f"train acc = {cand_acc:.1%}"
                    )

        # Keep top-K unique prompts
        seen: set = set()
        unique_candidates = []
        for score, p in sorted(candidates, reverse=True):
            if p not in seen:
                seen.add(p)
                unique_candidates.append((score, p))

        beam = unique_candidates[:beam_size]

        best_acc, best_prompt = beam[0]
        _log(f"\n  🏆 Best prompt this round — train acc: {best_acc:.1%}")
        _log(f"  Prompt preview: {best_prompt[:120]}…")

        history.append(
            {
                "round":        round_num,
                "best_prompt":  best_prompt,
                "train_acc":    best_acc,
                "beam":         [(a, p[:80]) for a, p in beam],
            }
        )

    # ── Final evaluation on held-out eval set ────────────────────────────────
    final_prompt       = beam[0][1]
    final_train_acc    = beam[0][0]
    eval_acc, eval_err = evaluate(final_prompt, eval_data)

    _log("\n" + "=" * 60)
    _log("  OPTIMIZATION COMPLETE")
    _log("=" * 60)
    _log(f"\n🎯 Final prompt:\n{final_prompt}\n")
    _log(f"📊 Train accuracy : {final_train_acc:.1%}")
    _log(f"📊 Eval  accuracy : {eval_acc:.1%}  ({len(eval_err)} errors on held-out set)")

    return {
        "best_prompt":         final_prompt,
        "best_train_accuracy": final_train_acc,
        "best_eval_accuracy":  eval_acc,
        "history":             history,
    }
