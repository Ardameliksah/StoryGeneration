"""Full pipeline: prompt → premise → outline → story.

Usage:
    python inference.py "A dog saves the day"
    python inference.py  # uses a built-in example

Works zero-shot (no fine-tuning needed). Automatically uses fine-tuned
checkpoints from models/t5_outline/ and models/bart_story/ when available.
"""

import pathlib
import sys

import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T5_CHECKPOINT   = "models/t5_outline"
BART_CHECKPOINT = "models/bart_story"
T5_FALLBACK     = "t5-small"
BART_FALLBACK   = "facebook/bart-base"

# Premise template slots — MoPS-style structured premise
_PREMISE_TEMPLATE = (
    "Title: {title}\n"
    "Setting: An everyday environment relevant to the title.\n"
    "Main character: A person affected by the situation in the title.\n"
    "Goal: To resolve or respond to the situation described.\n"
    "Conflict: An unexpected complication arises along the way."
)


# ---------------------------------------------------------------------------
# Stage 1 — Premise expansion (template-based, instant, no model required)
# ---------------------------------------------------------------------------

def expand_premise(title: str) -> str:
    return _PREMISE_TEMPLATE.format(title=title)


# ---------------------------------------------------------------------------
# Stage 2 — Outline generation (T5-small)
# ---------------------------------------------------------------------------

_outline_cache: tuple | None = None


def _load_outline_model() -> tuple[T5TokenizerFast, T5ForConditionalGeneration]:
    global _outline_cache
    if _outline_cache is not None:
        return _outline_cache
    checkpoint = T5_CHECKPOINT if pathlib.Path(T5_CHECKPOINT, "config.json").exists() else T5_FALLBACK
    source = "fine-tuned checkpoint" if checkpoint == T5_CHECKPOINT else "pre-trained (zero-shot)"
    print(f"[Outline model] Loading T5-small from {checkpoint} ({source}) ...")
    tok   = T5TokenizerFast.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(DEVICE)
    model.eval()
    _outline_cache = (tok, model)
    return _outline_cache


def generate_outline(title: str) -> str:
    tok, model = _load_outline_model()
    prompt = f"generate outline: {title}"
    ids    = tok(prompt, return_tensors="pt", max_length=64, truncation=True).input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Stage 3 — Story generation (BART-base)
# ---------------------------------------------------------------------------

_story_cache: tuple | None = None


def _load_story_model() -> tuple[BartTokenizerFast, BartForConditionalGeneration]:
    global _story_cache
    if _story_cache is not None:
        return _story_cache
    checkpoint = BART_CHECKPOINT if pathlib.Path(BART_CHECKPOINT, "config.json").exists() else BART_FALLBACK
    source = "fine-tuned checkpoint" if checkpoint == BART_CHECKPOINT else "pre-trained (zero-shot)"
    print(f"[Story model]   Loading BART-base from {checkpoint} ({source}) ...")
    tok   = BartTokenizerFast.from_pretrained(checkpoint)
    model = BartForConditionalGeneration.from_pretrained(checkpoint).to(DEVICE)
    model.eval()
    _story_cache = (tok, model)
    return _story_cache


def generate_story(title: str, outline: str) -> str:
    tok, model = _load_story_model()
    prompt = f"{title} outline: {outline}"
    ids    = tok(prompt, return_tensors="pt", max_length=128, truncation=True).input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(title: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Prompt: {title}")
    print("=" * 60)

    premise = expand_premise(title)
    print(f"\n[Stage 1 — Premise]\n{premise}")

    outline = generate_outline(title)
    print(f"\n[Stage 2 — Outline]\n{outline}")

    story = generate_story(title, outline)
    print(f"\n[Stage 3 — Story]\n{story}")
    print()

    return {"premise": premise, "outline": outline, "story": story}


if __name__ == "__main__":
    title = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "She finally found what she had been looking for"
    run_pipeline(title)
