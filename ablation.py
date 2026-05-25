"""Ablation study — measures the contribution of each pipeline component.

Two separate ablation suites:
  --data roc  (default)  ROC conditions + premise_trained
  --data wp              Full WP ablation on WritingPrompts models/data

ROC conditions
--------------
full            Baseline  : T5 event outline + DOME memory  (ROC models)
no_memory       -DOME     : event outline, no memory block
no_outline      -Outline  : title → BART only
with_premise    +Premise* : full + premise in BART input (inference-only, unfair)
sent_outline    Sent OL   : oracle sentence outline (upper bound)
wp_models       WP cross  : full pipeline with WP-trained models on WP val
premise_trained +Premise  : premise-TRAINED BART + premise at inference (fair test)

WP conditions  (--data wp)
--------------------------
full            WP full pipeline
no_memory       WP, no DOME memory
no_outline      WP, title → BART only
sent_outline    WP, oracle sentence outline (upper bound)

Usage
-----
    python ablation.py                          # ROC, 200 examples
    python ablation.py --n 50                   # quick smoke test
    python ablation.py --data wp --n 200        # full WP ablation
    python ablation.py --conditions full no_memory premise_trained
    python ablation.py --bert-model roberta-large
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
# Data configs
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

# Same reference data as ROC, but uses the premise-trained BART checkpoint
ROC_PREMISE_CFG = {
    "val":  "data/processed/story_val.jsonl",
    "test": "data/processed/story_test.jsonl",
    "t5":   "models/t5_outline",
    "bart": "models/bart_story_premise",   # trained with premise in input
}

_DATA_CFGS = {
    "roc":         ROC_CFG,
    "wp":          WP_CFG,
    "roc_premise": ROC_PREMISE_CFG,
}

# ---------------------------------------------------------------------------
# Condition registries
# ---------------------------------------------------------------------------

# Each entry: (label, kwargs for _run_condition, data_config_key)
ROC_ABLATION_CONDITIONS: list[tuple[str, dict, str]] = [
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
        "roc",       # inference-only: BART not trained with premise (unfair baseline)
    ),
    (
        "sent_outline",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=True),
        "roc",
    ),
    (
        "wp_models",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=False),
        "wp",        # cross-dataset: WP models evaluated on WP val data
    ),
    (
        "premise_trained",
        dict(use_memory=True,  use_outline=True,  use_premise=True,  use_sent_outline=False),
        "roc_premise",   # BART retrained with premise → fair test
    ),
]

WP_ABLATION_CONDITIONS: list[tuple[str, dict, str]] = [
    (
        "full",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=False),
        "wp",
    ),
    (
        "no_memory",
        dict(use_memory=False, use_outline=True,  use_premise=False, use_sent_outline=False),
        "wp",
    ),
    (
        "no_outline",
        dict(use_memory=False, use_outline=False, use_premise=False, use_sent_outline=False),
        "wp",
    ),
    (
        "sent_outline",
        dict(use_memory=True,  use_outline=True,  use_premise=False, use_sent_outline=True),
        "wp",
    ),
]

_ROC_NAMES = [c[0] for c in ROC_ABLATION_CONDITIONS]
_WP_NAMES  = [c[0] for c in WP_ABLATION_CONDITIONS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_examples(path: str, n: int) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))
    return examples


def _extract_title(input_field: str) -> str:
    """Strip ' outline: ...' and ' premise: ...' suffixes."""
    return input_field.split(" outline:")[0].split(" premise:")[0].strip()


def _sentence_outline(story: str) -> str:
    """Oracle sentence outline: first | middle | last sentence from reference."""
    sents = [s.strip() for s in _SENT_RE.split(story) if len(s.strip()) > 5]
    if len(sents) < 2:
        return story
    return f"{sents[0]} | {sents[len(sents) // 2]} | {sents[-1]}"


def _set_checkpoints(t5_path: str, bart_path: str) -> None:
    inference.T5_CHECKPOINT   = t5_path
    inference.BART_CHECKPOINT = bart_path
    inference._outline_cache  = None
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
    _set_checkpoints(t5_path, bart_path)
    hypotheses: list[str] = []
    references: list[str] = []

    for i, ex in enumerate(examples):
        title = _extract_title(ex["input"])
        ref   = ex["target"]

        outline = (
            ""                          if not use_outline else
            _sentence_outline(ref)      if use_sent_outline else
            generate_outline(title)
        )
        memory  = extract_memory(title) if use_memory  else ""
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
# Main ablation runner
# ---------------------------------------------------------------------------

def ablation(
    n: int                = 200,
    split: str            = "val",
    bert_model: str       = "distilbert-base-uncased",
    conditions: list[str] | None = None,
    data: str             = "roc",
) -> dict[str, dict[str, float]]:
    """Run one ablation suite and return results as a dict.

    Parameters
    ----------
    data : 'roc' runs the ROC conditions (incl. premise_trained);
           'wp'  runs the WritingPrompts conditions.
    """
    if data == "wp":
        active_registry = WP_ABLATION_CONDITIONS
        default_names   = _WP_NAMES
        suite_label     = "WritingPrompts"
    else:
        active_registry = ROC_ABLATION_CONDITIONS
        default_names   = _ROC_NAMES
        suite_label     = "ROCStories"

    active = conditions or default_names
    all_results: dict[str, dict[str, float]] = {}

    for label, kwargs, cfg_key in active_registry:
        if label not in active:
            continue

        cfg  = _DATA_CFGS[cfg_key]
        path = cfg[split]

        if not pathlib.Path(path).exists():
            print(f"\n[SKIP] {label}: data not found at {path}")
            continue
        if not pathlib.Path(cfg["t5"],   "config.json").exists():
            print(f"\n[SKIP] {label}: T5 checkpoint not found at {cfg['t5']}")
            continue
        if not pathlib.Path(cfg["bart"], "config.json").exists():
            print(f"\n[SKIP] {label}: BART checkpoint not found at {cfg['bart']}")
            continue

        print(f"\n── {suite_label} | {label} {'─'*(35-len(label))}")
        examples = _load_examples(path, n)
        hyps, refs = _run_condition(label, examples, cfg["t5"], cfg["bart"], **kwargs)

        print("  Computing metrics …")
        scores = metrics.compute_all(hyps, refs, bert_model=bert_model, verbose=True)
        all_results[label] = scores

    # ── Print comparison table ────────────────────────────────────────────
    if not all_results:
        print("\nNo results — make sure models and data are present.")
        return all_results

    W = 74
    print(f"\n\n{'='*W}")
    print(f"{'ABLATION RESULTS — ' + suite_label:^{W}}")
    print(f"{'n=' + str(n) + ' | split=' + split:^{W}}")
    print(f"{'='*W}")
    print(f"{'Condition':<18}  {'ROUGE-L':>9}  {'BLEU':>7}  {'METEOR':>8}  {'BERTScore':>10}")
    print("-" * W)

    col_keys = ["rouge_l", "bleu", "meteor", "bertscore"]
    col_best = {k: max(v[k] for v in all_results.values()) for k in col_keys}

    for label, scores in all_results.items():
        def fmt(k: str, fs: str) -> str:
            v = scores[k]
            s = format(v, fs)
            return f"*{s}*" if abs(v - col_best[k]) < 1e-6 else f" {s} "

        print(
            f"{label:<18}  "
            f"{fmt('rouge_l',  '.4f'):>10}  "
            f"{fmt('bleu',     '.2f'):>8}  "
            f"{fmt('meteor',   '.4f'):>9}  "
            f"{fmt('bertscore','.4f'):>11}"
        )

    print("=" * W)
    print("  * = best in column")

    if data == "roc":
        print()
        print("Condition legend:")
        print("  full            – T5 event outline + DOME memory (ROC models)")
        print("  no_memory       – DOME memory removed")
        print("  no_outline      – outline removed entirely")
        print("  with_premise    – premise at inference (BART not trained for it — unfair)")
        print("  sent_outline    – oracle sentence outline (upper bound)")
        print("  wp_models       – WP-trained models on WP val data (cross-dataset)")
        print("  premise_trained – premise-retrained BART + premise at inference (fair)")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",       choices=["roc", "wp"], default="roc",
        help="Which ablation suite to run: roc (default) or wp",
    )
    parser.add_argument(
        "--n",          type=int, default=200,
        help="Examples per condition (default: 200)",
    )
    parser.add_argument(
        "--split",      choices=["val", "test"], default="val",
    )
    parser.add_argument(
        "--bert-model", default="distilbert-base-uncased",
        help="HuggingFace model for BERTScore",
    )
    parser.add_argument(
        "--conditions", nargs="*",
        help="Run only these conditions (default: all for chosen --data)",
    )
    args = parser.parse_args()
    ablation(
        n=args.n,
        split=args.split,
        bert_model=args.bert_model,
        conditions=args.conditions,
        data=args.data,
    )
