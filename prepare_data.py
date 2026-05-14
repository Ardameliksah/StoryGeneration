"""Prepare ROCStories data from tab-separated txt files.

Input format (each line):  id<TAB>sentence1. sentence2. sentence3. sentence4. sentence5.
- Sentence 1 is used as the story prompt (no title column in this format).
- Event-sequence outline (EtriCA-style) = event(s1) [E2] event(s3) [E3] event(s5).
  Each event = subject + verb phrase (with aux/neg) + object + key prep phrase.
- Story target = all 5 sentences.

Train split: data/raw/rocstories_train.txt  → 90% train, 10% val
Test  split: data/raw/rocstories_test.txt   → test

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import json
import pathlib
import re
import random

import spacy

random.seed(42)

TRAIN_TXT  = "data/raw/rocstories_train.txt"
TEST_TXT   = "data/raw/rocstories_test.txt"
OUT_DIR    = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 512

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# Only load components needed for dependency parsing
_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])


def _span_text(token: spacy.tokens.Token) -> str:
    return " ".join(
        t.text for t in sorted(token.subtree, key=lambda x: x.i)
        if t.pos_ not in ("PUNCT", "SPACE") and t.dep_ not in ("punct", "cc", "mark")
    )


def _event_from_doc(doc: spacy.tokens.Doc, raw: str) -> str:
    """Extract an EtriCA-style event tuple from an already-parsed spaCy Doc."""
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


def split_sentences(text: str, n: int = 5) -> list[str]:
    """Split story text into exactly n sentences."""
    parts = _SENT_RE.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < n:
        parts = [p.strip() for p in text.strip().split(". ") if p.strip()]
    return parts[:n]


def load_txt(path: str) -> list[list[str]]:
    """Return a list of sentence-lists, one per story."""
    stories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sents = split_sentences(line)
            if len(sents) == 5:
                stories.append(sents)
    return stories


def write_jsonl(stories: list[list[str]], outline_path: pathlib.Path, story_path: pathlib.Path) -> None:
    # Batch-parse all outline sentences in one pass for speed
    n = len(stories)
    s1s = [s[0] for s in stories]
    s3s = [s[2] for s in stories]
    s5s = [s[4] for s in stories]

    all_sents = s1s + s3s + s5s
    docs = list(_nlp.pipe(all_sents, batch_size=BATCH_SIZE))
    docs_s1, docs_s3, docs_s5 = docs[:n], docs[n:2*n], docs[2*n:]

    with open(outline_path, "w", encoding="utf-8") as fo, \
         open(story_path,   "w", encoding="utf-8") as fs:
        for i, sents in enumerate(stories):
            s1, s2, s3, s4, s5 = sents
            prompt  = s1
            e1 = _event_from_doc(docs_s1[i], s1)
            e2 = _event_from_doc(docs_s3[i], s3)
            e3 = _event_from_doc(docs_s5[i], s5)
            outline = f"{e1} | {e2} | {e3}"
            story   = " ".join(sents)
            fo.write(json.dumps({"input": prompt, "target": outline}, ensure_ascii=False) + "\n")
            fs.write(json.dumps({"input": f"{prompt} outline: {outline}", "target": story}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print(f"Loading {TRAIN_TXT} ...")
    train_stories = load_txt(TRAIN_TXT)
    random.shuffle(train_stories)
    cut = int(0.9 * len(train_stories))
    train_split, val_split = train_stories[:cut], train_stories[cut:]

    write_jsonl(train_split, OUT_DIR / "outline_train.jsonl", OUT_DIR / "story_train.jsonl")
    print(f"  train: {len(train_split)} examples")

    write_jsonl(val_split, OUT_DIR / "outline_val.jsonl", OUT_DIR / "story_val.jsonl")
    print(f"  val:   {len(val_split)} examples")

    print(f"Loading {TEST_TXT} ...")
    test_stories = load_txt(TEST_TXT)
    write_jsonl(test_stories, OUT_DIR / "outline_test.jsonl", OUT_DIR / "story_test.jsonl")
    print(f"  test:  {len(test_stories)} examples")

    print("Done.")
