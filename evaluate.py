"""Evaluate the fine-tuned pipeline on the validation set.

Metrics: ROUGE-L and BLEU.
Usage:
    python evaluate.py                          # ROCStories, 500 val examples
    python evaluate.py --data wp --n 100        # WritingPrompts, 100 examples
    python evaluate.py --split test             # use test split
"""

import argparse
import json

import sacrebleu
from rouge_score import rouge_scorer as rs

import inference
from inference import generate_outline, generate_story

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
    return input_field.split(" outline:")[0].strip()


def evaluate(data: str = "roc", split: str = "val", n: int = 500) -> None:
    cfg = DATA_CONFIG[data]

    # Point inference module at the right checkpoints
    inference.T5_CHECKPOINT   = cfg["t5"]
    inference.BART_CHECKPOINT = cfg["bart"]
    inference._outline_cache  = None
    inference._story_cache    = None

    print(f"Evaluating {n} examples from {cfg[split]} ...")
    examples = load_examples(cfg[split], n)

    scorer     = rs.RougeScorer(["rougeL"], use_stemmer=True)
    hypotheses = []
    references = []

    for i, ex in enumerate(examples):
        title   = extract_title(ex["input"])
        outline = generate_outline(title)
        pred    = generate_story(title, outline)
        hypotheses.append(pred)
        references.append(ex["target"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(examples)} done ...")

    rouge_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"\n{'='*40}")
    print(f"Dataset : {data.upper()}")
    print(f"ROUGE-L : {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"BLEU    : {bleu.score:.2f}")
    print(f"{'='*40}")
    print("Note: low BLEU/ROUGE is expected for creative story generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  choices=["roc", "wp"], default="roc",
                        help="Which model/dataset to evaluate (default: roc)")
    parser.add_argument("--n",     type=int, default=500,  help="Number of examples")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(data=args.data, split=args.split, n=args.n)
