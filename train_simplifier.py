import argparse
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


MODEL_NAME = "t5-small"
PREFIX = "Simplify this medical text into plain English: "
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 128
DEFAULT_TRAIN_PATH = Path("data/simplification_train.csv")
DEFAULT_VAL_PATH = Path("data/simplification_val.csv")
FALLBACK_TRAIN_PATH = Path("data/processed/cochrane_simplification_train.csv")
FALLBACK_VAL_PATH = Path("data/processed/cochrane_simplification_validation.csv")
DEFAULT_MODEL_DIR = Path("models/simplifier")
TARGET_PREFIX_PATTERNS = [
    r"^in this review[,:\s]+",
    r"^we found that\s+",
    r"^we found\s+",
    r"^this review found that\s+",
    r"^this review found\s+",
    r"^the review found that\s+",
    r"^the review found\s+",
    r"^review authors found that\s+",
    r"^review authors found\s+",
    r"^the findings of this review (were|are)\s+",
    r"^evidence in this review (is|was)\s+(current to [^.]+\.\s+)?",
]
BAD_PHRASE_PATTERNS = [
    r"\bin this review\b",
    r"\bwe found\b",
    r"\bthis review\b",
    r"\bsystematic review\b",
]
REVIEW_SUMMARY_PATTERNS = [
    r"\bsystematic review\b",
    r"\bsearch(es|ed)?\b",
    r"\brandomi[sz]ed (clinical )?trial\b",
    r"\bparticipants\b",
    r"\bstudies\b",
    r"\bwe included\b",
    r"\bthis review included\b",
    r"\bthe review included\b",
]


def ensure_training_dependencies() -> None:
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Training requires the 'accelerate' package. Install it with:\n"
            "pip install 'accelerate>=0.26.0'"
        ) from exc


def resolve_dataset_paths(
    train_path: str = str(DEFAULT_TRAIN_PATH),
    val_path: str = str(DEFAULT_VAL_PATH),
) -> Tuple[Path, Path]:
    resolved_train = Path(train_path)
    resolved_val = Path(val_path)

    if not resolved_train.exists() and FALLBACK_TRAIN_PATH.exists():
        resolved_train = FALLBACK_TRAIN_PATH
    if not resolved_val.exists() and FALLBACK_VAL_PATH.exists():
        resolved_val = FALLBACK_VAL_PATH

    missing = [str(path) for path in (resolved_train, resolved_val) if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Required simplification CSV file(s) not found: "
            f"{joined}. Expected columns: original, reference."
        )

    return resolved_train, resolved_val


def load_csv_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    required_columns = {"original", "reference"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    df = df.dropna(subset=["original", "reference"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty after dropping rows with missing values.")

    df = df[["original", "reference"]].copy()
    return Dataset.from_pandas(df, preserve_index=False)


def clean_target_text(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    for pattern in TARGET_PREFIX_PATTERNS:
        updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        if updated != cleaned:
            cleaned = updated.strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def contains_bad_phrase(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in BAD_PHRASE_PATTERNS)


def looks_like_review_summary(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(("we searched", "we included", "this review included", "the searches")):
        return True
    return sum(
        bool(re.search(pattern, text, flags=re.IGNORECASE))
        for pattern in REVIEW_SUMMARY_PATTERNS
    ) >= 2


def is_nearly_identical(original: str, reference: str) -> bool:
    original_norm = " ".join(original.lower().split())
    reference_norm = " ".join(reference.lower().split())
    return SequenceMatcher(None, original_norm, reference_norm).ratio() >= 0.92


def inspect_reference_quality(df: pd.DataFrame, dataset_label: str) -> None:
    print(f"\nInspecting {dataset_label} references")
    for index, reference in enumerate(df["reference"].head(20), start=1):
        print(f"{index:02d}. {reference[:300]}")

    total_rows = len(df)
    for phrase in ["In this review", "we found", "this review", "systematic review"]:
        count = df["reference"].fillna("").str.contains(re.escape(phrase), case=False, regex=True).sum()
        print(f"{phrase}: {count} / {total_rows} ({count / total_rows:.1%})")


def clean_and_filter_dataframe(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    before_rows = len(df)
    working = df[["original", "reference"]].copy()
    working["original"] = working["original"].astype(str).map(lambda text: " ".join(text.strip().split()))
    working["reference"] = working["reference"].astype(str).map(lambda text: " ".join(text.strip().split()))
    working["reference"] = working["reference"].map(clean_target_text)

    empty_mask = (working["original"] == "") | (working["reference"] == "")
    identical_mask = working.apply(
        lambda row: is_nearly_identical(row["original"], row["reference"]), axis=1
    )
    bad_phrase_mask = working["reference"].map(contains_bad_phrase)
    review_mask = working["reference"].map(looks_like_review_summary) | bad_phrase_mask

    filtered = working[~empty_mask & ~identical_mask & ~review_mask].drop_duplicates().reset_index(drop=True)

    print(f"\n{dataset_label} cleaning summary")
    print(f"Rows before filtering: {before_rows}")
    print(f"Dropped empty rows: {int(empty_mask.sum())}")
    print(f"Dropped near-identical rows: {int(identical_mask.sum())}")
    print(f"Dropped bad-phrase rows: {int(bad_phrase_mask.sum())}")
    print(f"Dropped review-style rows: {int(review_mask.sum())}")
    print(f"Rows after filtering: {len(filtered)}")
    print("Sample kept examples:")
    for index, row in filtered.head(5).iterrows():
        print(f"- original: {row['original'][:180]}")
        print(f"  reference: {row['reference'][:180]}")

    return filtered


def preprocess_dataset(dataset: Dataset, tokenizer: T5Tokenizer) -> Dataset:
    if len(dataset) == 0:
        return dataset

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        inputs = [PREFIX + text.strip() for text in batch["original"]]
        targets = [clean_target_text(text) for text in batch["reference"]]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_OUTPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )

        masked_labels = []
        for sequence in labels["input_ids"]:
            masked_labels.append(
                [token if token != tokenizer.pad_token_id else -100 for token in sequence]
            )
        model_inputs["labels"] = masked_labels
        return model_inputs

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def build_trainer(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    output_dir: Path,
) -> Seq2SeqTrainer:
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        predict_with_generate=True,
        eval_strategy="epoch" if len(eval_dataset) > 0 else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


def train(
    train_path: str = str(DEFAULT_TRAIN_PATH),
    val_path: str = str(DEFAULT_VAL_PATH),
    output_dir: str = str(DEFAULT_MODEL_DIR),
) -> None:
    ensure_training_dependencies()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_train_path, resolved_val_path = resolve_dataset_paths(train_path, val_path)
    train_df = pd.read_csv(resolved_train_path)
    val_df = pd.read_csv(resolved_val_path)
    inspect_reference_quality(train_df, "training")
    train_cleaned_df = clean_and_filter_dataframe(train_df, "training")
    val_cleaned_df = clean_and_filter_dataframe(val_df, "validation")
    train_raw_dataset = Dataset.from_pandas(train_cleaned_df, preserve_index=False)
    eval_raw_dataset = Dataset.from_pandas(val_cleaned_df, preserve_index=False)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    train_dataset = preprocess_dataset(train_raw_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_raw_dataset, tokenizer)

    trainer = build_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        output_dir=output_path,
    )

    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def load_model_and_tokenizer(model_dir: str = str(DEFAULT_MODEL_DIR)):
    model_path = Path(model_dir)
    source = model_path if model_path.exists() else MODEL_NAME
    tokenizer = T5Tokenizer.from_pretrained(str(source))
    model = T5ForConditionalGeneration.from_pretrained(str(source))
    model.eval()
    return tokenizer, model


def simplify(text: str, model_dir: str = str(DEFAULT_MODEL_DIR)) -> str:
    tokenizer, model = load_model_and_tokenizer(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(
        PREFIX + text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generated = model.generate(
        **encoded,
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a T5 medical text simplifier.")
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="Path to the training CSV with 'original' and 'reference' columns.",
    )
    parser.add_argument(
        "--val-path",
        default=str(DEFAULT_VAL_PATH),
        help="Path to the validation CSV with 'original' and 'reference' columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--demo-text",
        default=None,
        help="Optional text to simplify after training or using an existing model.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run simplification with an existing saved model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        train(train_path=args.train_path, val_path=args.val_path, output_dir=args.output_dir)

    if args.demo_text:
        result = simplify(args.demo_text, model_dir=args.output_dir)
        print(f"Input: {args.demo_text}")
        print(f"Simplified: {result}")


if __name__ == "__main__":
    main()
