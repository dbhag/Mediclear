import argparse
import os

import pandas as pd
import sacrebleu
from sklearn.metrics import accuracy_score, f1_score

from mediclear.neural_pipeline import MediClearNeuralPipeline

try:
    from easse.sari import corpus_sari
    HAS_EASSE = True
except:
    HAS_EASSE = False


DEFAULT_SIMPLIFIER_DIR = "models/t5_simplifier"
DEFAULT_CLASSIFIER_DIR = "models/pubhealth_classifier"
DEFAULT_SIMPLIFICATION_CSV = "data/cochrane_simplification_test.csv"
DEFAULT_HEALTH_FACT_CSV = "data/health_fact_test.csv"


def normalize_label(x):
    return str(x).strip().lower()


def make_results_folder():
    if not os.path.exists("results"):
        os.makedirs("results")


def evaluate_simplification(pipeline, csv_file):
    df = pd.read_csv(csv_file)

    if "original" not in df.columns or "reference" not in df.columns:
        raise ValueError("Simplification file must contain 'original' and 'reference' columns")

    originals = df["original"].fillna("").tolist()
    references = df["reference"].fillna("").tolist()

    predictions = []
    for text in originals:
        pred = pipeline.simplify_text(text)
        predictions.append(pred)

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    sari = None
    if HAS_EASSE:
        sari = corpus_sari(
            orig_sents=originals,
            sys_sents=predictions,
            refs_sents=[references]
        )

    outputs_df = pd.DataFrame({
        "original": originals,
        "reference": references,
        "prediction": predictions
    })
    outputs_df.to_csv("results/neural_simplification_outputs.csv", index=False)

    metrics = {
        "Model": "Neural Simplifier",
        "BLEU": round(bleu, 2),
        "SARI": round(sari, 2) if sari is not None else None
    }

    pd.DataFrame([metrics]).to_csv("results/neural_simplification_metrics.csv", index=False)

    return metrics


def evaluate_health_fact(pipeline, csv_file):
    df = pd.read_csv(csv_file)

    if "claim" not in df.columns or "label" not in df.columns:
        raise ValueError("Health fact file must contain 'claim' and 'label' columns")

    claims = df["claim"].fillna("").tolist()

    true_labels = []
    for x in df["label"].tolist():
        true_labels.append(normalize_label(x))

    predicted_labels = []
    confidences = []

    for claim in claims:
        pred_label, confidence, _ = pipeline.classify_credibility(claim)
        predicted_labels.append(normalize_label(pred_label))
        confidences.append(round(float(confidence), 4))

    acc = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

    outputs_df = pd.DataFrame({
        "claim": claims,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "confidence": confidences
    })
    outputs_df.to_csv("results/neural_health_fact_outputs.csv", index=False)

    metrics = {
        "Model": "Neural Health Fact Classifier",
        "Accuracy": round(acc, 2),
        "Macro_F1": round(macro_f1, 2)
    }

    pd.DataFrame([metrics]).to_csv("results/neural_health_fact_metrics.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simplifier_dir",
        default=DEFAULT_SIMPLIFIER_DIR,
        help="Folder that contains the trained simplifier model"
    )

    parser.add_argument(
        "--classifier_dir",
        default=DEFAULT_CLASSIFIER_DIR,
        help="Folder that contains the trained classifier model"
    )

    parser.add_argument(
        "--simplification_csv",
        default=DEFAULT_SIMPLIFICATION_CSV,
        help="Test CSV for simplification"
    )

    parser.add_argument(
        "--health_fact_csv",
        default=DEFAULT_HEALTH_FACT_CSV,
        help="Test CSV for health fact classification"
    )

    args = parser.parse_args()

    make_results_folder()

    pipeline = MediClearNeuralPipeline(
        simplifier_dir=args.simplifier_dir,
        classifier_dir=args.classifier_dir
    )

    simplification_metrics = evaluate_simplification(
        pipeline,
        args.simplification_csv
    )

    health_fact_metrics = evaluate_health_fact(
        pipeline,
        args.health_fact_csv
    )

    print("\n=== Simplification Metrics ===")
    print(pd.DataFrame([simplification_metrics]).to_string(index=False))

    print("\n=== Health Fact Metrics ===")
    print(pd.DataFrame([health_fact_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()