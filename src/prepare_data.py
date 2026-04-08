import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from datasets import DatasetDict, load_dataset


SIMPLIFICATION_DATASET = "GEM/cochrane-simplification"
PUBHEALTH_DATASET = "ImperialCollegeLondon/health_fact"
PUBHEALTH_ALLOWED_LABELS = {"true", "false", "mixture"}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def inspect_dataset(name: str, dataset: DatasetDict) -> None:
    print(f"\n{name}")
    for split_name, split_dataset in dataset.items():
        print(
            f"  split={split_name} rows={len(split_dataset)} "
            f"columns={list(split_dataset.column_names)}"
        )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"  wrote {len(df)} rows -> {output_path}")


def clean_simplification_split(split_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame(
        {
            "original": split_df["source"].map(normalize_text),
            "reference": split_df["target"].map(normalize_text),
        }
    )
    cleaned = cleaned[(cleaned["original"] != "") & (cleaned["reference"] != "")]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def clean_pubhealth_split(split_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame(
        {
            "text": split_df["claim"].map(normalize_text),
            "label": split_df["label"].map(lambda value: normalize_text(value).lower()),
        }
    )
    cleaned = cleaned[(cleaned["text"] != "") & (cleaned["label"] != "")]
    cleaned = cleaned[cleaned["label"].isin(PUBHEALTH_ALLOWED_LABELS)]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def prepare_simplification(output_dir: Path) -> Dict[str, Path]:
    dataset = load_dataset(SIMPLIFICATION_DATASET)
    inspect_dataset(SIMPLIFICATION_DATASET, dataset)

    written_files = {}
    for split_name, split_dataset in dataset.items():
        split_df = split_dataset.to_pandas()
        cleaned = clean_simplification_split(split_df)
        output_path = output_dir / f"cochrane_simplification_{split_name}.csv"
        export_csv(cleaned, output_path)
        written_files[split_name] = output_path
    return written_files


def prepare_pubhealth(output_dir: Path) -> Dict[str, Path]:
    dataset = load_dataset(PUBHEALTH_DATASET, trust_remote_code=True)
    inspect_dataset(PUBHEALTH_DATASET, dataset)

    written_files = {}
    for split_name, split_dataset in dataset.items():
        split_df = split_dataset.to_pandas()
        cleaned = clean_pubhealth_split(split_df)
        output_path = output_dir / f"pubhealth_{split_name}.csv"
        export_csv(cleaned, output_path)
        written_files[split_name] = output_path
    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, inspect, clean, and export medical NLP datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where cleaned CSV files will be written.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "simplification", "pubhealth"],
        default="all",
        help="Prepare one dataset family or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.only in {"all", "simplification"}:
        print("Preparing simplification dataset...")
        prepare_simplification(output_dir)

    if args.only in {"all", "pubhealth"}:
        print("\nPreparing PUBHEALTH dataset...")
        prepare_pubhealth(output_dir)


if __name__ == "__main__":
    main()
