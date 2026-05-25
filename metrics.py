"""Shared evaluation metrics for story generation.

Provides ROUGE-L, BLEU, METEOR, and BERTScore in one place so both
evaluate.py and ablation.py can import them without code duplication.

Requires:
    pip install sacrebleu rouge-score nltk bert-score
"""

import nltk
import sacrebleu
from rouge_score import rouge_scorer as rs

# Download required NLTK resources once (silent if already present)
for _pkg in ("wordnet", "punkt_tab", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level ROUGE-L F-measure (macro-average over examples)."""
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def bleu(hypotheses: list[str], references: list[str]) -> float:
    """Corpus-level BLEU score via sacrebleu."""
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def meteor(hypotheses: list[str], references: list[str]) -> float:
    """Average METEOR score (NLTK implementation, word-level)."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tok = nltk.word_tokenize(hyp.lower())
        ref_tok = nltk.word_tokenize(ref.lower())
        scores.append(
            nltk.translate.meteor_score.single_meteor_score(ref_tok, hyp_tok)
        )
    return sum(scores) / len(scores) if scores else 0.0


def bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
) -> float:
    """Average BERTScore F1.

    Default: distilbert-base-uncased — fast on Colab T4 (~2 min for 200 ex.)
    Higher accuracy: pass model_type='roberta-large' (~5 min for 200 ex.)
    """
    from bert_score import score as _bs  # lazy import — heavy dependency

    _, _, F1 = _bs(hypotheses, references, model_type=model_type, verbose=False)
    return F1.mean().item()


# ---------------------------------------------------------------------------
# Convenience: compute all four at once
# ---------------------------------------------------------------------------

def compute_all(
    hypotheses: list[str],
    references: list[str],
    bert_model: str = "distilbert-base-uncased",
    verbose: bool = True,
) -> dict[str, float]:
    """Compute ROUGE-L, BLEU, METEOR, and BERTScore F1. Returns a dict."""
    if verbose:
        print("  ROUGE-L   … ", end="", flush=True)
    r = rouge_l(hypotheses, references)
    if verbose:
        print(f"{r:.4f}")

    if verbose:
        print("  BLEU      … ", end="", flush=True)
    b = bleu(hypotheses, references)
    if verbose:
        print(f"{b:.2f}")

    if verbose:
        print("  METEOR    … ", end="", flush=True)
    m = meteor(hypotheses, references)
    if verbose:
        print(f"{m:.4f}")

    if verbose:
        print("  BERTScore … ", end="", flush=True)
    bs = bertscore(hypotheses, references, model_type=bert_model)
    if verbose:
        print(f"{bs:.4f}")

    return {"rouge_l": r, "bleu": b, "meteor": m, "bertscore": bs}
