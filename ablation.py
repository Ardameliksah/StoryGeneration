"""Ablation study — measures the contribution of each pipeline component.

Six conditions are evaluated side-by-side across four metrics.

Conditions
----------
full          Baseline  : T5 event outline + DOME memory  (ROC models, ROC val)
no_memory     -DOME     : event outline, no memory block
no_outline    -Outline  : title → BART only (no outline, no memory)
with_premise  +Premise  : full + structured premise text in BART input
sent_outline  Sent OL   : oracle sentence outline (begin|mid|end from reference)
wp_models     WP Data   : full pipeline with WritingPrompts-trained models (WP val)

The 'sent_outline' condition is an oracle upper-bound: it feeds actual
sentences from the reference story as the outline, showing the maximum gain
possible from a perfect outline generator.

Usage
-----
    python ablation.py                   # 200 examples, val split
    python ablation.py --n 50            # quick smoke test (~5 min on T4)
    python ablation.py --n 200 --split test
    python ablation.py --bert-model roberta-large  # higher-quality BERTScore
    python ablation.py --conditions full no_memory wp_models  # subset
"""

import argparse
import json
import pathlib
import re

import inference
from inference import (
    generate_outline, generate_story,
    extract_memory, expand_premise,
)
import metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

ROC_CFG = {
    "val":  "data/processed/story_val.jsonl",
    "test": "data/processed/story_test.jsonl",
    "t5":   "models/t5_outline",
    "bart": "models/bart_story",
}

WP_CFG = {
    "val":  "data/processed/wp_story_val.jsonl",
    "test": "data/processed/wp_story_test.jsonl",
    "t5":   "models/t5_outline_wp",
    "bart": "models/bart_story_wp",
}


def _load_examples(path: str, n: int) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))
    return examples


def _extract_title(input_field: str) -> str:
    """Strip ' outline: ...' suffix added during data prep."""
    return input_field.split(" outline:")[0].strip()


def _sentence_outline(story: str) -> str:
    """Oracle sentence outline: first | middle | last sentence from reference.

    Used in the 'sent_outline' condition to show the upper bound of what a
    perfect outline generator could achieve.
    """
    sents = [s.strip() for s in _SENT_RE.split(story) if len(s.strip()) > 5]
    if len(sents) < 2:
        return story
    s_begin = sents[0]
    s_mid   = sents[len(sents) // 2]
    s_end   = sents[-1]
    return f"{s_begin} | {s_mid} | {s_end}"


def _set_checkpoints(t5_path: str, bart_path: str) -> None:
    """Hot-swap the model checkpoints used by the inference module."""
    inference.T5_CHECKPOINT   = t5_path
    inference.BART_CHECKPOINT = bart_path
    inference._outline_cache  = None   # force reload
    inference._story_cache    = None


# ---------------------------------------------------------------------------
# Per-condition generation
# ---------------------------------------------------------------------------

def _run_condition(
    label: str,
    examples: list[dict],
    t5_path: str,
    bart_path: str,
    use_memory: bool       = True,
    use_outline: bool      = True,
    use_premise: bool      = False,
    use_sent_outline: bool = False,
) -> tuple[list[str], list[str]]:
    """Generate stories for all examples under one ablation condition."""
    _set_checkpoints(t5_path, bart_path)
    hypotheses: list[str] = []
    references: list[str] = []

    for i, ex in enumerate(examples):
        title = _extract_title(ex["input"])
        ref   = ex["target"]

        # ── Choose outline ────────────────────────────────────────────────
        if not use_outline:
            outline = ""
        elif use_sent_outline:
            outline = _sentence_outline(ref)    # oracle from reference
        else:
            outline = generate_outline(title)   # T5-generated event outline

        # ── Memory ───────────────────────────────────────────────────────
        memory = extract_memory(title) if use_memory else ""

        # ── Premise ──────────────────────────────────────────────────────
        premise = expand_premise(title) if use_premise else ""

        pred = generate_story(
            title, outline, memory,
            premise=premise,
            use_premise=use_premise,
            use_memory=use_memory,
            use_outline=use_outline,
        )
        hypotheses.append(pred)
        references.append(ref)

        if (i + 1) % 50 == 0:
            print(f"    [{label}] {i+1}/{len(examples)} done …")

    return hypotheses, references


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

# Each entry: (name, kwargs for _run_condition, data_config_key)
ABLATION_CONDITIONS: list[tuple[str, dict, str]] = [
    (
        "full",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=False),
        "roc",
    ),
    (
        "no_memory",
        dict(use_memory=False, use_outline=True,  use_premise=False, use_sent_outline=False),
        "roc",
    ),
    (
        "no_outline",
        dict(use_memory=False, use_outline=False, use_premise=False, use_sent_outline=False),
        "roc",
    ),
    (
        "with_premise",
        dict(use_memory=True,  use_outline=True,  use_premise=True,  use_sent_outline=False),
        "roc",
    ),
    (
        "sent_outline",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=True),
        "roc",
    ),
    (
        "wp_models",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=False),
        "wp",
    ),
]

_CONDITION_NAMES = [c[0] for c in ABLATION_CONDITIONS]


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def ablation(
    n: int                = 200,
    split: str            = "val",
    bert_model: str       = "distilbert-base-uncased",
    conditions: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Run the ablation study and print a comparison table.

    Parameters
    ----------
    n          : number of evaluation examples per condition
    split      : 'val' or 'test'
    bert_model : HuggingFace model ID for BERTScore
    conditions : subset of condition names to run (default: all)
    """
    data_cfgs = {"roc": ROC_CFG, "wp": WP_CFG}
    active = conditions or _CONDITION_NAMES

    all_results: dict[str, dict[str, float]] = {}

    for label, kwargs, data_key in ABLATION_CONDITIONS:
        if label not in active:
            continue

        cfg  = data_cfgs[data_key]
        path = cfg[split]

        # Skip gracefully if data / model checkpoints are missing
        if not pathlib.Path(path).exists():
            print(f"\n[SKIP] {label}: data not found at {path}")
            continue
        if not pathlib.Path(cfg["t5"], "config.json").exists():
            print(f"\n[SKIP] {label}: T5 checkpoint not found at {cfg['t5']}")
            continue
        if not pathlib.Path(cfg["bart"], "config.json").exists():
            print(f"\n[SKIP] {label}: BART checkpoint not found at {cfg['bart']}")
            continue

        print(f"\n── Condition: {label} {'─'*(40-len(label))}")
        examples = _load_examples(path, n)
        hyps, refs = _run_condition(label, examples, cfg["t5"], cfg["bart"], **kwargs)

        print("  Computing metrics …")
        scores = metrics.compute_all(hyps, refs, bert_model=bert_model, verbose=True)
        all_results[label] = scores

    # ── Print comparison table ────────────────────────────────────────────
    if not all_results:
        print("\nNo results — make sure models and data are present.")
        return all_results

    W = 72
    print(f"\n\n{'='*W}")
    print(f"{'ABLATION RESULTS':^{W}}")
    print(f"{'n=' + str(n) + ' | split=' + split:^{W}}")
    print(f"{'='*W}")
    print(f"{'Condition':<16}  {'ROUGE-L':>9}  {'BLEU':>7}  {'METEOR':>8}  {'BERTScore':>10}")
    print("-" * W)

    # Bold the best score in each column (ASCII marker)
    col_keys = ["rouge_l", "bleu", "meteor", "bertscore"]
    col_best = {k: max(v[k] for v in all_results.values()) for k in col_keys}

    for label, scores in all_results.items():
        def fmt(k: str, fmt_str: str) -> str:
            val = scores[k]
            s   = format(val, fmt_str)
            return f"*{s}*" if abs(val - col_best[k]) < 1e-6 else f" {s} "

        print(
            f"{label:<16}  "
            f"{fmt('rouge_l',  '.4f'):>10}  "
            f"{fmt('bleu',     '.2f'):>8}  "
            f"{fmt('meteor',   '.4f'):>9}  "
            f"{fmt('bertscore','.4f'):>11}"
        )

    print("=" * W)
    print("  * = best in column")
    print()
    print("Condition legend:")
    print("  full         – T5 event outline + DOME memory (ROC models)")
    print("  no_memory    – DOME memory module removed")
    print("  no_outline   – no outline fed to story model")
    print("  with_premise – structured premise prepended to BART input")
    print("  sent_outline – oracle sentence outline (upper bound)")
    print("  wp_models    – WritingPrompts-trained models on WP val data")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",          type=int, default=200,
        help="Examples per condition (default: 200)",
    )
    parser.add_argument(
        "--split",      choices=["val", "test"], default="val",
    )
    parser.add_argument(
        "--bert-model", default="distilbert-base-uncased",
        help="HuggingFace model ID for BERTScore "
             "(default: distilbert-base-uncased; roberta-large for accuracy)",
    )
    parser.add_argument(
        "--conditions", nargs="*", choices=_CONDITION_NAMES,
        help="Run only these conditions (default: all)",
    )
    args = parser.parse_args()
    ablation(
        n=args.n,
        split=args.split,
        bert_model=args.bert_model,
        conditions=args.conditions,
    )
