import argparse
import os

import pandas as pd
from datasets import load_dataset


SIMPLIFICATION_DATASET = "GEM/cochrane-simplification"
HEALTH_FACT_DATASET = "ImperialCollegeLondon/health_fact"

OUTPUT_DIR = "data"
ALLOWED_LABELS = ["false", "mixture", "true", "unproven"]


def clean_text(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def clean_target(x):
    if isinstance(x, list):
        if len(x) > 0:
            return clean_text(x[0])
        return ""
    return clean_text(x)


def clean_list(x):
    if x is None:
        return ""
    if isinstance(x, list):
        new_list = []
        for item in x:
            item = clean_text(item)
            if item != "":
                new_list.append(item)
        return " | ".join(new_list)
    return clean_text(x)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv(df, file_path):
    folder = os.path.dirname(file_path)
    if folder != "":
        make_folder(folder)
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} ({len(df)} rows)")


def prepare_simplification(output_dir):
    print(f"\nLoading {SIMPLIFICATION_DATASET}...")
    dataset = load_dataset(SIMPLIFICATION_DATASET)

    for split_name in dataset:
        print(f"\nWorking on simplification split: {split_name}")

        split_data = dataset[split_name]
        df = split_data.to_pandas()

        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError("source or target column is missing")

        raw_rows = len(df)

        original_list = []
        reference_list = []

        for i in range(len(df)):
            original = clean_text(df.loc[i, "source"])
            reference = clean_target(df.loc[i, "target"])

            if original != "" and reference != "":
                original_list.append(original)
                reference_list.append(reference)

        new_df = pd.DataFrame({
            "original": original_list,
            "reference": reference_list
        })

        after_empty = len(new_df)

        new_df = new_df.drop_duplicates()
        new_df = new_df.reset_index(drop=True)

        final_rows = len(new_df)

        print(
            f"  simplification cleaning: raw={raw_rows}, "
            f"after_empty_removal={after_empty}, final={final_rows}"
        )

        file_path = os.path.join(output_dir, f"cochrane_simplification_{split_name}.csv")
        save_csv(new_df, file_path)


def prepare_health_fact(output_dir):
    print(f"\nLoading {HEALTH_FACT_DATASET}...")
    dataset = load_dataset(HEALTH_FACT_DATASET, trust_remote_code=True)

    for split_name in dataset:
        print(f"\nWorking on health_fact split: {split_name}")

        split_data = dataset[split_name]
        df = split_data.to_pandas()

        needed = ["claim", "explanation", "main_text", "label"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"{col} column is missing")

        label_names = split_data.features["label"].names

        raw_rows = len(df)

        rows = []

        for i in range(len(df)):
            claim = clean_text(df.loc[i, "claim"])
            explanation = clean_text(df.loc[i, "explanation"])
            main_text = clean_text(df.loc[i, "main_text"])

            label_value = df.loc[i, "label"]
            label = ""

            try:
                label_index = int(label_value)
                if label_index >= 0 and label_index < len(label_names):
                    label = clean_text(label_names[label_index]).lower()
            except:
                label = clean_text(label_value).lower()

            if claim == "" or label == "":
                continue

            if label not in ALLOWED_LABELS:
                continue

            row = {
                "claim": claim,
                "explanation": explanation,
                "main_text": main_text,
                "label": label
            }

            if "sources" in df.columns:
                row["sources"] = clean_list(df.loc[i, "sources"])

            if "subjects" in df.columns:
                row["subjects"] = clean_list(df.loc[i, "subjects"])

            if "date_published" in df.columns:
                row["date_published"] = clean_text(df.loc[i, "date_published"])

            if "fact_checkers" in df.columns:
                row["fact_checkers"] = clean_list(df.loc[i, "fact_checkers"])

            if "claim_id" in df.columns:
                row["claim_id"] = clean_text(df.loc[i, "claim_id"])

            rows.append(row)

        after_filter = len(rows)

        new_df = pd.DataFrame(rows)
        new_df = new_df.drop_duplicates()
        new_df = new_df.reset_index(drop=True)

        final_rows = len(new_df)

        print(
            f"  health_fact cleaning: raw={raw_rows}, "
            f"after_filter={after_filter}, final={final_rows}"
        )

        file_path = os.path.join(output_dir, f"health_fact_{split_name}.csv")
        save_csv(new_df, file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--only",
        choices=["all", "simplification", "health_fact"],
        default="all"
    )

    args = parser.parse_args()

    print("Starting dataset preparation...")
    print(f"Saving files to: {args.output_dir}")
    print(f"Mode: {args.only}")

    if args.only == "all" or args.only == "simplification":
        prepare_simplification(args.output_dir)

    if args.only == "all" or args.only == "health_fact":
        prepare_health_fact(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()