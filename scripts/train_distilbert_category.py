"""
Fine-tune DistilBERT for 9-class financial complaint category classification.

Model:  distilbert-base-uncased
Data:   SQLite complaints + data/supplementary_train_categories.json (701 samples)
Split:  80% train / 20% validation (stratified)
Saves:  models/distilbert_category/  (HuggingFace format)

Estimated time: ~20-30 min on CPU.

Usage:
    python scripts/train_distilbert_category.py
"""
import sys
import json
import sqlite3
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data.database import DB_PATH

MODELS_DIR   = Path("models")
MODEL_OUT    = MODELS_DIR / "distilbert_category"
SUPP_FILE    = "data/supplementary_train_categories.json"
EVAL_FILE    = "data/eval/test_categories.json"
BASE_MODEL   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
LR           = 2e-5
MAX_EPOCHS   = 8
PATIENCE     = 2   # early stopping

LABELS = [
    "Fraud & Unauthorized Charges", "Billing Concern", "Transaction Dispute",
    "Account Access Issue", "Card Services", "Reward & Points Issue",
    "Loan & Credit Issue", "Customer Service Complaint", "General Inquiry",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #
def load_all_data():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT anonymized_text, category FROM complaints "
        "WHERE category IS NOT NULL AND anonymized_text IS NOT NULL"
    ).fetchall()
    conn.close()
    texts  = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    if Path(SUPP_FILE).exists():
        with open(SUPP_FILE) as f:
            for s in json.load(f):
                texts.append(s["text"])
                labels.append(s["label"])

    return texts, labels


def load_eval_data():
    with open(EVAL_FILE) as f:
        samples = json.load(f)
    return [s["text"] for s in samples], [s["label"] for s in samples]


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ------------------------------------------------------------------ #
# Class weights (handle imbalance)                                     #
# ------------------------------------------------------------------ #
def compute_class_weights(labels: list[str]) -> torch.Tensor:
    counts = Counter(labels)
    total  = len(labels)
    weights = [total / (len(LABELS) * counts.get(l, 1)) for l in LABELS]
    return torch.tensor(weights, dtype=torch.float)


# ------------------------------------------------------------------ #
# Train / eval loops                                                   #
# ------------------------------------------------------------------ #
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main():
    print("=" * 60)
    print("DistilBERT Category Classifier Fine-tuning")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("Estimated time on CPU: ~20-30 minutes\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_all, y_all = load_all_data()
    print(f"Total training samples: {len(X_all)}")
    dist = Counter(y_all)
    for lbl in LABELS:
        print(f"  {lbl:<40} {dist.get(lbl, 0):>4}")

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

    # Tokenizer and datasets
    tokenizer  = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    train_ds   = ComplaintDataset(X_train, y_train, tokenizer)
    val_ds     = ComplaintDataset(X_val,   y_val,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Optimizer, scheduler, weighted loss
    total_steps = len(train_loader) * MAX_EPOCHS
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    class_weights = compute_class_weights(y_train).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with early stopping
    best_val_acc  = 0.0
    patience_ctr  = 0
    best_epoch    = 0

    print("\n--- Training ---")
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t0        = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        elapsed   = time.time() - t0
        marker    = " ← best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  "
              f"({elapsed:.0f}s){marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_ctr = 0
            # Save best model
            MODELS_DIR.mkdir(exist_ok=True)
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    total_time = time.time() - t_start
    print(f"\nBest val accuracy: {best_val_acc*100:.1f}% at epoch {best_epoch}")
    print(f"Total training time: {total_time/60:.1f} min")

    # Final evaluation on 45-sample hand-labelled test set
    print("\n--- Evaluation on hand-labelled test set ---")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_OUT).to(device)
    model.eval()

    X_test, y_test = load_eval_data()
    enc = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
    y_pred = [ID2LABEL[i] for i in logits.argmax(dim=-1).cpu().tolist()]

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.1f}%  (TF-IDF+LR baseline: 84.4%,  zero-shot: 64.4%)")
    print()
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
