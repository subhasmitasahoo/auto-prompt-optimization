"""
Entry point for the Customer Support Ticket — Auto Prompt Optimizer.

Usage
-----
# Run with defaults (3 rounds, beam size 3)
python main.py

# Customise
python main.py --rounds 5 --beam 4 --feedbacks 3 --steps 2

# Dry-run: just evaluate the initial prompt, no optimization
python main.py --dry-run

Environment variable required:
    ANTHROPIC_API_KEY   Your Anthropic API key
"""

import argparse
import json
import os
import sys
import time

from data      import TRAINING_DATA, EVAL_DATA
from optimizer import evaluate, run_apo
from prompts   import INITIAL_PROMPT


def check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "❌  ANTHROPIC_API_KEY is not set.\n"
            "    Export it before running:\n\n"
            "      export ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto Prompt Optimization for customer support ticket classification."
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of APO optimization rounds (default: 3)",
    )
    parser.add_argument(
        "--beam", type=int, default=3,
        help="Beam size — how many top prompts to keep each round (default: 3)",
    )
    parser.add_argument(
        "--feedbacks", type=int, default=3,
        help="Gradient feedbacks (reasons) requested per prompt (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=2,
        help="New prompt candidates generated per gradient reason (default: 2)",
    )
    parser.add_argument(
        "--max-train-sample", type=int, default=30,
        help="Max training examples evaluated per round (default: 30, use 0 for all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only evaluate the initial prompt; skip optimization.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save the final result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    check_api_key()
    args = parse_args()

    max_sample = len(TRAINING_DATA) if args.max_train_sample == 0 else args.max_train_sample

    print(f"\n{'='*60}")
    print("  Customer Support Ticket — Auto Prompt Optimizer")
    print(f"{'='*60}")
    print(f"  Training examples : {len(TRAINING_DATA)}")
    print(f"  Eval examples     : {len(EVAL_DATA)}")
    print(f"  Train sample/round: {max_sample}")

    if args.dry_run:
        print("\n⚡ Dry-run mode — evaluating initial prompt only.\n")
        train_acc, train_errors = evaluate(INITIAL_PROMPT, TRAINING_DATA)
        eval_acc,  eval_errors  = evaluate(INITIAL_PROMPT, EVAL_DATA)
        print(f"📊 Initial prompt train accuracy : {train_acc:.1%}")
        print(f"📊 Initial prompt eval  accuracy : {eval_acc:.1%}")
        if eval_errors:
            print(f"\nEval errors ({len(eval_errors)}):")
            for err in eval_errors:
                print(f"  ✗  [{err['expected']}] predicted as [{err['predicted']}]")
                print(f"     \"{err['text'][:80]}…\"")
        return

    print(f"  Rounds            : {args.rounds}")
    print(f"  Beam size         : {args.beam}")
    print(f"  Feedbacks/prompt  : {args.feedbacks}")
    print(f"  Candidates/grad   : {args.steps}")
    print()

    t0     = time.time()
    result = run_apo(
        initial_prompt      = INITIAL_PROMPT,
        train_data          = TRAINING_DATA,
        eval_data           = EVAL_DATA,
        num_rounds          = args.rounds,
        beam_size           = args.beam,
        steps_per_gradient  = args.steps,
        num_feedbacks       = args.feedbacks,
        max_train_sample    = max_sample,
        verbose             = True,
    )
    elapsed = time.time() - t0

    print(f"\n⏱  Total time: {elapsed:.0f}s")
    print(f"\n{'='*60}")
    print("  FINAL OPTIMIZED PROMPT")
    print(f"{'='*60}")
    print(result["best_prompt"])
    print(f"\n📊 Train accuracy: {result['best_train_accuracy']:.1%}")
    print(f"📊 Eval  accuracy: {result['best_eval_accuracy']:.1%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
