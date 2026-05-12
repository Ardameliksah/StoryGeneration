"""Fine-tune T5-small: storytitle → pseudo-outline (sentence1 + sentence3 + sentence5)."""

import json
import pathlib

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup

MODEL_NAME = "t5-small"
TRAIN_FILE = "data/processed/outline_train.jsonl"
VAL_FILE   = "data/processed/outline_val.jsonl"
OUTPUT_DIR = "models/t5_outline"
MAX_IN     = 64
MAX_OUT    = 128
BATCH      = 16
EPOCHS     = 3
LR         = 2e-5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class OutlineDataset(Dataset):
    def __init__(self, path: str, tokenizer: T5TokenizerFast) -> None:
        self.data = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.tok  = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        # T5 uses task prefix in the input
        inp = self.tok(
            f"generate outline: {row['input']}",
            max_length=MAX_IN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tok(
            row["target"],
            max_length=MAX_OUT,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tgt.input_ids.squeeze().clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids":      inp.input_ids.squeeze(),
            "attention_mask": inp.attention_mask.squeeze(),
            "labels":         labels,
        }


def evaluate(model: T5ForConditionalGeneration, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
    return total_loss / len(loader)


def train() -> None:
    print(f"Device: {DEVICE}")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    # num_workers=0 required on Windows
    train_loader = DataLoader(OutlineDataset(TRAIN_FILE, tokenizer), batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(OutlineDataset(VAL_FILE,   tokenizer), batch_size=BATCH, shuffle=False, num_workers=0)

    total_steps = len(train_loader) * EPOCHS
    optimizer   = AdamW(model.parameters(), lr=LR)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 200 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(train_loader)} | loss {running_loss/step:.4f}")

        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch} complete | val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
