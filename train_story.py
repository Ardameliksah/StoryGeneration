"""Fine-tune BART-base: (prompt + event-sequence outline) → full story.

Supports ROCStories and WritingPrompts. Designed for epoch-by-epoch training
on Colab — each session trains --epochs epochs then saves full state to Drive.

Usage:
    # ROCStories (default)
    python train_story.py

    # WritingPrompts, one epoch at a time
    python train_story.py --data wp --epochs 1
    python train_story.py --data wp --epochs 1 --resume   # next session
    python train_story.py --data wp --epochs 1 --resume   # next session
"""

import argparse
import json
import pathlib

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizerFast, get_linear_schedule_with_warmup

MODEL_NAME = "facebook/bart-base"
MAX_IN     = 192
MAX_OUT    = 256
LR         = 2e-5
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

DATA_CONFIG = {
    "roc": {
        "train": "data/processed/story_train.jsonl",
        "val":   "data/processed/story_val.jsonl",
        "out":   "models/bart_story",
        "batch": 8,
    },
    "wp": {
        "train": "data/processed/wp_story_train.jsonl",
        "val":   "data/processed/wp_story_val.jsonl",
        "out":   "models/bart_story_wp",
        "batch": 4,  # reduced for longer WP sequences
    },
}


class StoryDataset(Dataset):
    def __init__(self, path: str, tokenizer: BartTokenizerFast) -> None:
        self.data = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.tok  = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        inp = self.tok(
            row["input"],
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


def evaluate(model: BartForConditionalGeneration, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
    return total_loss / len(loader)


def train(args: argparse.Namespace) -> None:
    cfg        = DATA_CONFIG[args.data]
    output_dir = pathlib.Path(cfg["out"])
    state_file = output_dir / "training_state.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE} | Dataset: {args.data}")

    # ── Load model ──────────────────────────────────────────────────────────
    if args.resume and state_file.exists():
        print(f"Resuming from {output_dir} ...")
        tokenizer     = BartTokenizerFast.from_pretrained(output_dir)
        model         = BartForConditionalGeneration.from_pretrained(output_dir).to(DEVICE)
        state         = torch.load(state_file, map_location=DEVICE)
        start_epoch   = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        total_steps   = state["total_steps"]
        print(f"  Resuming from epoch {start_epoch} | best_val_loss={best_val_loss:.4f}")
    else:
        tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
        model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        start_epoch   = 1
        best_val_loss = float("inf")
        total_steps   = None

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        StoryDataset(cfg["train"], tokenizer),
        batch_size=cfg["batch"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        StoryDataset(cfg["val"], tokenizer),
        batch_size=cfg["batch"], shuffle=False, num_workers=0,
    )

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    if total_steps is None:
        total_steps = len(train_loader) * 3  # assume 3-epoch plan for LR schedule

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    if args.resume and state_file.exists():
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    # ── Training loop ────────────────────────────────────────────────────────
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
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
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Best model saved to {output_dir}")

        # Save training state after every epoch so Colab can resume
        torch.save({
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "total_steps":   total_steps,
        }, state_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   choices=["roc", "wp"], default="roc",
                        help="Dataset to train on (default: roc)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Epochs to train this session (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved checkpoint")
    train(parser.parse_args())
