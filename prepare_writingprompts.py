"""Prepare WritingPrompts data from HuggingFace.

Downloads euclaise/writingprompts and processes it into the same JSONL format
used by train_outline.py and train_story.py.

Stories are truncated to MAX_WORDS words. Outline = EtriCA-style event sequence
extracted from beginning, middle, and end of the truncated story.

Output files:
    data/processed/wp_outline_{train,val,test}.jsonl
    data/processed/wp_story_{train,val,test}.jsonl

Usage:
    python prepare_writingprompts.py                     # full dataset (~272K)
    python prepare_writingprompts.py --max-stories 50000 # faster subset

Requires:
    pip install datasets spacy
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import pathlib
import random
import re

import spacy
from datasets import load_dataset

random.seed(42)

MAX_WORDS  = 150
BATCH_SIZE = 512
OUT_DIR    = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_SENT_RE  = re.compile(r'(?<=[.!?])\s+')
_CLEAN_RE = re.compile(r'<[^>]+>|\[.*?\]|\*{1,3}|#{1,6}\s?|_{2,}')

_PROFANITY = re.compile(
    r'\b(fuck|shit|cunt|bitch|asshole|bastard|motherfuck)\w*\b', re.IGNORECASE
)


def is_clean(text: str) -> bool:
    return not _PROFANITY.search(text)

_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])


def clean_text(text: str) -> str:
    text = _CLEAN_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _span_text(token: spacy.tokens.Token) -> str:
    return " ".join(
        t.text for t in sorted(token.subtree, key=lambda x: x.i)
        if t.pos_ not in ("PUNCT", "SPACE") and t.dep_ not in ("punct", "cc", "mark")
    )


def _event_from_doc(doc: spacy.tokens.Doc, raw: str) -> str:
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return " ".join(raw.split()[:10])

    subj_token = next((t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")), None)
    subj = _span_text(subj_token) if subj_token else ""

    aux_tokens = [t.text for t in root.children if t.dep_ in ("aux", "auxpass", "neg")]
    verb_phrase = " ".join(aux_tokens + [root.text])

    obj_token = next((t for t in root.children if t.dep_ in ("dobj", "attr", "xcomp", "oprd")), None)
    obj = _span_text(obj_token) if obj_token else ""

    prep_token = next((t for t in root.children if t.dep_ == "prep"), None)
    prep = _span_text(prep_token) if prep_token else ""
    if len(prep.split()) > 6:
        prep = " ".join(prep.split()[:6])

    parts = [p for p in [subj, verb_phrase, obj, prep] if p]
    result = " ".join(parts).strip()
    return result if result else " ".join(raw.split()[:10])


def process_and_write(raw_data: list[dict], outline_path: pathlib.Path, story_path: pathlib.Path) -> int:
    # First pass: clean and filter
    items = []
    for ex in raw_data:
        prompt = clean_text(ex["prompt"])
        story  = clean_text(ex["story"])
        if not prompt or not story:
            continue
        if not is_clean(prompt) or not is_clean(story):
            continue
        story = " ".join(story.split()[:MAX_WORDS])
        sents = split_sentences(story)
        if len(sents) < 3:
            continue
        items.append({
            "prompt":   prompt,
            "story":    story,
            "s_begin":  sents[0],
            "s_mid":    sents[len(sents) // 2],
            "s_end":    sents[-1],
        })

    # Batch-parse all outline sentences at once
    n = len(items)
    all_sents = [it["s_begin"] for it in items] + \
                [it["s_mid"]   for it in items] + \
                [it["s_end"]   for it in items]
    docs = list(_nlp.pipe(all_sents, batch_size=BATCH_SIZE))
    docs_begin, docs_mid, docs_end = docs[:n], docs[n:2*n], docs[2*n:]

    with open(outline_path, "w", encoding="utf-8") as fo, \
         open(story_path,   "w", encoding="utf-8") as fs:
        for i, it in enumerate(items):
            e1 = _event_from_doc(docs_begin[i], it["s_begin"])
            e2 = _event_from_doc(docs_mid[i],   it["s_mid"])
            e3 = _event_from_doc(docs_end[i],   it["s_end"])
            outline = f"{e1} | {e2} | {e3}"
            fo.write(json.dumps({"input": it["prompt"], "target": outline},                      ensure_ascii=False) + "\n")
            fs.write(json.dumps({"input": f"{it['prompt']} outline: {outline}",
                                  "target": it["story"]}, ensure_ascii=False) + "\n")
    return len(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Limit total stories for faster processing (default: all)")
    args = parser.parse_args()

    print("Downloading WritingPrompts from HuggingFace ...")
    dataset = load_dataset("euclaise/writingprompts", split="train")
    data = [{"prompt": ex["prompt"], "story": ex["story"]} for ex in dataset]

    random.shuffle(data)
    if args.max_stories:
        data = data[:args.max_stories]

    n       = len(data)
    t_end   = int(0.90 * n)
    v_end   = int(0.95 * n)
    splits  = [("train", data[:t_end]), ("val", data[t_end:v_end]), ("test", data[v_end:])]

    print(f"Total: {n} | Train: {t_end} | Val: {v_end - t_end} | Test: {n - v_end}")

    for name, subset in splits:
        print(f"Processing {name} ...")
        count = process_and_write(
            subset,
            OUT_DIR / f"wp_outline_{name}.jsonl",
            OUT_DIR / f"wp_story_{name}.jsonl",
        )
        print(f"  {name}: {count} examples written")

    print("Done.")
