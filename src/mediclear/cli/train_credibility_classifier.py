import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
TRAIN_FILE = "data/health_fact_train.csv"
VAL_FILE = "data/health_fact_validation.csv"
OUTPUT_DIR = "models/pubhealth_classifier"

LABEL_TO_ID = {
    "false": 0,
    "mixture": 1,
    "true": 2,
    "unproven": 3,
}

ID_TO_LABEL = {
    0: "false",
    1: "mixture",
    2: "true",
    3: "unproven",
}


def clean_text(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def normalize_label(x):
    return str(x).strip().lower()


def load_csv_dataset(path):
    df = pd.read_csv(path)

    if "claim" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'claim' and 'label' columns")

    rows = []

    for i in range(len(df)):
        claim = clean_text(df.loc[i, "claim"])
        label = normalize_label(df.loc[i, "label"])

        if claim == "":
            continue

        if label not in LABEL_TO_ID:
            continue

        main_text = ""
        if "main_text" in df.columns:
            main_text = clean_text(df.loc[i, "main_text"])

        if main_text != "":
            full_text = claim + " [SEP] " + main_text
        else:
            full_text = claim

        rows.append({
            "text": full_text,
            "label": LABEL_TO_ID[label]
        })

    new_df = pd.DataFrame(rows)

    if len(new_df) == 0:
        raise ValueError(f"{path} has no valid rows after cleaning")

    new_df = new_df.drop_duplicates()
    new_df = new_df.reset_index(drop=True)

    return Dataset.from_pandas(new_df)


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    pred_ids = predictions.argmax(axis=1)

    correct = 0
    for i in range(len(labels)):
        if pred_ids[i] == labels[i]:
            correct += 1

    if len(labels) == 0:
        accuracy = 0.0
    else:
        accuracy = correct / len(labels)

    return {
        "accuracy": accuracy
    }


def save_label_map(output_dir):
    label_map = {
        "id2label": {
            "0": "false",
            "1": "mixture",
            "2": "true",
            "3": "unproven"
        },
        "label2id": {
            "false": 0,
            "mixture": 1,
            "true": 2,
            "unproven": 3
        }
    }

    file_path = os.path.join(output_dir, "label_map.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)


def train_model(train_file, val_file, output_dir):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    print("Loading datasets...")
    train_ds = load_csv_dataset(train_file)
    val_ds = load_csv_dataset(val_file)

    print(f"Training rows: {len(train_ds)}")
    print(f"Validation rows: {len(val_ds)}")

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )

    tokenized_val = val_ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_label_map(output_dir)

    print(f"Saved classifier model to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=TRAIN_FILE)
    parser.add_argument("--val_file", default=VAL_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    train_model(args.train_file, args.val_file, args.output_dir)


if __name__ == "__main__":
    main()