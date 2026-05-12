"""Prepare ROCStories data from tab-separated txt files.

Input format (each line):  id<TAB>sentence1. sentence2. sentence3. sentence4. sentence5.
- Sentence 1 is used as the story prompt (no title column in this format).
- Pseudo-outline = sentence1 + sentence3 + sentence5 (beginning, middle, end).
- Story target   = all 5 sentences.

Train split: data/raw/rocstories_train.txt  → 90% train, 10% val
Test  split: data/raw/rocstories_test.txt   → test
"""

import json
import pathlib
import re
import random

random.seed(42)

TRAIN_TXT = "data/raw/rocstories_train.txt"
TEST_TXT  = "data/raw/rocstories_test.txt"
OUT_DIR   = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str, n: int = 5) -> list[str]:
    """Split story text into exactly n sentences."""
    parts = _SENT_RE.split(text.strip())
    # Strip trailing punctuation remnants from each part
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < n:
        # Fallback: split on period only
        parts = [p.strip() for p in text.strip().split(". ") if p.strip()]
    return parts[:n]


def load_txt(path: str) -> list[list[str]]:
    """Return a list of sentence-lists, one per story.
    Each line is a full story (5 sentences concatenated) with no ID column.
    """
    stories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sents = split_sentences(line)
            if len(sents) == 5:
                stories.append(sents)
            # skip malformed rows silently
    return stories


def write_jsonl(stories: list[list[str]], outline_path: pathlib.Path, story_path: pathlib.Path) -> None:
    with open(outline_path, "w", encoding="utf-8") as fo, \
         open(story_path,   "w", encoding="utf-8") as fs:
        for sents in stories:
            s1, s2, s3, s4, s5 = sents
            prompt  = s1                          # first sentence as the prompt
            outline = f"{s1} {s3} {s5}"           # beginning, middle, end
            story   = " ".join(sents)
            fo.write(json.dumps({"input": prompt, "target": outline}, ensure_ascii=False) + "\n")
            fs.write(json.dumps({"input": f"{prompt} outline: {outline}", "target": story}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # --- Train / val from train txt ---
    print(f"Loading {TRAIN_TXT} ...")
    train_stories = load_txt(TRAIN_TXT)
    random.shuffle(train_stories)
    cut = int(0.9 * len(train_stories))
    train_split, val_split = train_stories[:cut], train_stories[cut:]

    write_jsonl(train_split, OUT_DIR / "outline_train.jsonl", OUT_DIR / "story_train.jsonl")
    print(f"  train: {len(train_split)} examples")

    write_jsonl(val_split, OUT_DIR / "outline_val.jsonl", OUT_DIR / "story_val.jsonl")
    print(f"  val:   {len(val_split)} examples")

    # --- Test from test txt ---
    print(f"Loading {TEST_TXT} ...")
    test_stories = load_txt(TEST_TXT)
    write_jsonl(test_stories, OUT_DIR / "outline_test.jsonl", OUT_DIR / "story_test.jsonl")
    print(f"  test:  {len(test_stories)} examples")

    print("Done.")
