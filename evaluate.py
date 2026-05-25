"""Evaluate the fine-tuned pipeline on the validation / test set.

Metrics: ROUGE-L, BLEU, METEOR, BERTScore.

Usage:
    python evaluate.py                              # ROCStories, 500 val examples
    python evaluate.py --data wp --n 100            # WritingPrompts, 100 examples
    python evaluate.py --split test                 # use test split
    python evaluate.py --no-memory                  # disable DOME memory
    python evaluate.py --bert-model roberta-large   # higher-quality BERTScore
"""

import argparse
import json

import inference
from inference import generate_outline, generate_story, extract_memory
import metrics

DATA_CONFIG = {
    "roc": {
        "val":  "data/processed/story_val.jsonl",
        "test": "data/processed/story_test.jsonl",
        "t5":   "models/t5_outline",
        "bart": "models/bart_story",
    },
    "wp": {
        "val":  "data/processed/wp_story_val.jsonl",
        "test": "data/processed/wp_story_test.jsonl",
        "t5":   "models/t5_outline_wp",
        "bart": "models/bart_story_wp",
    },
}


def load_examples(path: str, n: int) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))
    return examples


def extract_title(input_field: str) -> str:
    """Strip the ' outline: ...' suffix that was added during data prep."""
    return input_field.split(" outline:")[0].strip()


def evaluate(
    data: str = "roc",
    split: str = "val",
    n: int = 500,
    use_memory: bool = True,
    bert_model: str = "distilbert-base-uncased",
) -> dict[str, float]:
    cfg = DATA_CONFIG[data]

    # Point inference module at the correct checkpoints
    inference.T5_CHECKPOINT   = cfg["t5"]
    inference.BART_CHECKPOINT = cfg["bart"]
    inference._outline_cache  = None
    inference._story_cache    = None

    print(f"\nEvaluating {n} examples from {cfg[split]}")
    print(f"  Memory module : {'ON' if use_memory else 'OFF'}")
    print(f"  BERTScore model: {bert_model}")
    examples = load_examples(cfg[split], n)

    hypotheses: list[str] = []
    references: list[str] = []

    for i, ex in enumerate(examples):
        title   = extract_title(ex["input"])
        outline = generate_outline(title)
        memory  = extract_memory(title) if use_memory else ""
        pred    = generate_story(title, outline, memory)
        hypotheses.append(pred)
        references.append(ex["target"])
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{len(examples)} stories …")

    print("\nComputing metrics:")
    scores = metrics.compute_all(hypotheses, references,
                                 bert_model=bert_model, verbose=True)

    print(f"\n{'='*48}")
    print(f"  Dataset   : {data.upper()} | split={split} | n={n}")
    print(f"  Memory    : {'on' if use_memory else 'off'}")
    print(f"{'='*48}")
    print(f"  ROUGE-L   : {scores['rouge_l']:.4f}")
    print(f"  BLEU      : {scores['bleu']:.2f}")
    print(f"  METEOR    : {scores['meteor']:.4f}")
    print(f"  BERTScore : {scores['bertscore']:.4f}")
    print(f"{'='*48}")
    print("Note: low BLEU/ROUGE is normal for open-ended story generation.")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       choices=["roc", "wp"], default="roc",
                        help="Dataset / model to evaluate (default: roc)")
    parser.add_argument("--n",          type=int, default=500,
                        help="Number of validation examples (default: 500)")
    parser.add_argument("--split",      choices=["val", "test"], default="val",
                        help="Evaluation split (default: val)")
    parser.add_argument("--no-memory",  action="store_true",
                        help="Disable the DOME memory module")
    parser.add_argument("--bert-model", default="distilbert-base-uncased",
                        help="HuggingFace model ID for BERTScore "
                             "(default: distilbert-base-uncased; "
                             "use roberta-large for higher accuracy)")
    args = parser.parse_args()

    evaluate(
        data=args.data,
        split=args.split,
        n=args.n,
        use_memory=not args.no_memory,
        bert_model=args.bert_model,
    )
