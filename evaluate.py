"""Evaluate the fine-tuned pipeline on the validation set.

Metrics: ROUGE-L and BLEU.
Usage:
    python evaluate.py                    # evaluates 500 examples from story_val.jsonl
    python evaluate.py --n 100            # evaluates 100 examples (faster)
    python evaluate.py --split test       # uses story_test.jsonl
"""

import argparse
import json

import sacrebleu
from rouge_score import rouge_scorer as rs

from inference import generate_outline, generate_story


def load_examples(split: str, n: int) -> list[dict]:
    path = f"data/processed/story_{split}.jsonl"
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))
    return examples


def extract_title(input_field: str) -> str:
    # Story dataset input format: "title outline: ..."
    return input_field.split(" outline:")[0].strip()


def evaluate(split: str = "val", n: int = 500) -> None:
    print(f"Evaluating {n} examples from story_{split}.jsonl ...")
    examples = load_examples(split, n)

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
    print(f"ROUGE-L : {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"BLEU    : {bleu.score:.2f}")
    print(f"{'='*40}")
    print("Note: low BLEU/ROUGE is expected for creative story generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=500,  help="Number of examples to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(split=args.split, n=args.n)
