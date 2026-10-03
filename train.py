"""
Sentiment Analysis using BERT (HuggingFace Transformers)
Fine-tunes bert-base-uncased on the Amazon product reviews dataset.
"""

import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
OUTPUT_DIR   = "./model"
NUM_LABELS   = 2          # 0 = negative, 1 = positive
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5

# ── Load dataset ─────────────────────────────────────────────────────────────
# 40,000 training reviews, 10,000 test reviews
print("Loading dataset...")
raw = load_dataset("fancyzhx/amazon_polarity", split={"train": "train[:40000]", "test": "test[:10000]"})

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["content"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

print("Tokenizing...")
tokenized = raw.map(tokenize, batched=True, batch_size=512)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ── Training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nFinal evaluation:")
preds_output = trainer.predict(tokenized["test"])
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

# ── Save ──────────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to '{OUTPUT_DIR}/'")
