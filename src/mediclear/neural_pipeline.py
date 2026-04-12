import json
import os
import re
from difflib import SequenceMatcher

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


SIMPLIFIER_DIR = "models/t5_simplifier"
CLASSIFIER_DIR = "models/pubhealth_classifier"
SIMPLIFY_PREFIX = "simplify: "

DEFAULT_LABELS = {
    0: "false",
    1: "mixture",
    2: "true",
    3: "unproven",
}

TERM_REPLACEMENTS = [
    ("myocardial infarction", "heart attack"),
    ("myocardial infarction symptoms", "heart attack symptoms"),
    ("experiencing myocardial infarction symptoms", "having heart attack symptoms"),
    ("should seek immediate intervention", "should get medical help right away"),
    ("immediate intervention", "medical help right away"),

    ("hypertension", "high blood pressure"),
    ("sodium intake", "salt intake"),
    ("sodium", "salt"),
    ("reduce sodium intake", "reduce salt intake"),

    ("metastatic disease", "cancer that has spread"),
    ("metastasis", "spread of cancer"),
    ("malignant", "cancerous"),
    ("benign", "not cancer"),
    ("carcinoma", "cancer"),

    ("dyspnea", "trouble breathing"),
    ("shortness of breath", "trouble breathing"),
    ("edema", "swelling"),
    ("intravenous", "through a vein"),
    ("oral administration", "taking by mouth"),
    ("analgesic", "pain medicine"),
    ("prognosis", "expected outcome"),

    ("diabetes mellitus", "diabetes"),
    ("blood glucose levels", "blood sugar levels"),
    ("monitor their blood glucose levels regularly", "check their blood sugar regularly"),

    ("asthma triggers", "things that trigger asthma"),
    ("avoid triggers", "avoid things that make symptoms worse"),
    ("pollen", "plant dust"),
    ("influenza", "the flu"),
    ("influenza infection", "the flu"),

    ("antibiotics", "infection medicine"),
    ("physician", "doctor"),
    ("prescribed", "told to take"),
    ("cardiovascular disease", "heart disease"),
    ("cholesterol", "fat in the blood"),

    ("stroke", "brain attack"),
    ("symptom of stroke", "sign of a stroke"),
    ("dehydration", "not having enough water in the body"),
    ("severe dehydration", "serious water loss"),

    ("physical activity", "exercise"),
    ("improve heart health", "help the heart stay healthy"),
    ("mental health", "emotional well-being"),

    ("vaccinated", "given a vaccine"),
    ("vaccination", "getting a vaccine"),
    ("prevent disease", "stop disease"),
    ("prevent diseases", "stop diseases"),

    ("pneumonia", "a serious lung infection"),
    ("cures cancer instantly", "cannot cure cancer right away"),
    ("cures cancer", "cannot cure cancer"),
    ("completely treat pneumonia", "fully cure pneumonia"),
]


class MediClearNeuralPipeline:
    def __init__(self, simplifier_dir=SIMPLIFIER_DIR, classifier_dir=CLASSIFIER_DIR):
        self.simplifier_dir = simplifier_dir
        self.classifier_dir = classifier_dir

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if not os.path.exists(self.simplifier_dir):
            raise FileNotFoundError(f"Missing simplifier model folder: {self.simplifier_dir}")

        if not os.path.exists(self.classifier_dir):
            raise FileNotFoundError(f"Missing classifier model folder: {self.classifier_dir}")

        self.simplifier_tokenizer = T5Tokenizer.from_pretrained(self.simplifier_dir)
        self.simplifier_model = T5ForConditionalGeneration.from_pretrained(self.simplifier_dir)
        self.simplifier_model.to(self.device)
        self.simplifier_model.eval()

        self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_dir)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_dir)
        self.classifier_model.to(self.device)
        self.classifier_model.eval()

        self.label_map = self.load_label_map()

    def normalize_label_name(self, label):
        label = str(label).strip().lower()

        if label == "label_0":
            return "false"
        if label == "label_1":
            return "mixture"
        if label == "label_2":
            return "true"
        if label == "label_3":
            return "unproven"

        return label

    def load_label_map(self):
        label_map_path = os.path.join(self.classifier_dir, "label_map.json")

        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "id2label" in data:
                cleaned = {}

                for key in data["id2label"]:
                    try:
                        new_key = int(key)
                        new_value = self.normalize_label_name(data["id2label"][key])
                        cleaned[new_key] = new_value
                    except Exception:
                        pass

                if len(cleaned) > 0:
                    return cleaned

        config_map = getattr(self.classifier_model.config, "id2label", None)

        if config_map:
            cleaned = {}

            for key in config_map:
                try:
                    new_key = int(key)
                    new_value = self.normalize_label_name(config_map[key])
                    cleaned[new_key] = new_value
                except Exception:
                    pass

            if len(cleaned) > 0:
                return cleaned

        return DEFAULT_LABELS.copy()

    def clean_text(self, text):
        text = str(text).strip()
        text = " ".join(text.split())
        return text

    def rule_based_simplify(self, text):
        simplified = self.clean_text(text)

        for old_term, new_term in TERM_REPLACEMENTS:
            simplified = re.sub(
                rf"\b{re.escape(old_term)}\b",
                new_term,
                simplified,
                flags=re.IGNORECASE,
            )

        if len(simplified) > 0:
            simplified = simplified[0].upper() + simplified[1:]

        return simplified

    def needs_fallback(self, original, simplified):
        original_clean = self.clean_text(original).lower()
        simplified_clean = self.clean_text(simplified).lower()

        if simplified_clean == "":
            return True

        ratio = SequenceMatcher(None, original_clean, simplified_clean).ratio()
        if ratio >= 0.92:
            return True

        for old_term, new_term in TERM_REPLACEMENTS:
            if old_term in original_clean and new_term not in simplified_clean:
                return True

        return False

    def clean_simplified_output(self, text):
        text = self.clean_text(text)

        lower = text.lower()
        if lower.startswith("simple:"):
            text = text[7:].strip()
        elif lower.startswith("simplified:"):
            text = text[11:].strip()

        return self.clean_text(text)
        
    def simplify_text(self, text, max_input_length=256, max_output_length=128):
        text = self.clean_text(text)

        if text == "":
            return ""

        prompt = SIMPLIFY_PREFIX + text

        inputs = self.simplifier_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.simplifier_model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True,
            )

        result = self.simplifier_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        result = self.clean_simplified_output(result)

        if self.needs_fallback(text, result):
            fallback = self.rule_based_simplify(text)
            fallback = self.clean_simplified_output(fallback)

            if fallback != "" and fallback.lower() != text.lower():
                return fallback

        return result

    def classify_credibility(self, text, max_length=256):
        text = self.clean_text(text)

        if text == "":
            return "unproven", 0.0, {}

        inputs = self.classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

        if pred_id in self.label_map:
            label = self.label_map[pred_id]
        else:
            label = f"class_{pred_id}"

        prob_map = {}

        for i in range(len(probs)):
            if i in self.label_map:
                label_name = self.label_map[i]
            else:
                label_name = f"class_{i}"

            prob_map[label_name] = float(probs[i].item())

        return label, confidence, prob_map

    def extract_term_changes(self, original, simplified):
        original_lower = self.clean_text(original).lower()
        simplified_lower = self.clean_text(simplified).lower()

        changes = []
        seen = set()

        for old_term, new_term in TERM_REPLACEMENTS:
            if old_term in original_lower and new_term in simplified_lower:
                pair = (old_term, new_term)
                if pair not in seen:
                    changes.append(pair)
                    seen.add(pair)

        if len(changes) > 0:
            return changes

        original_words = re.findall(r"\b[\w-]+\b", original_lower)
        simplified_words = re.findall(r"\b[\w-]+\b", simplified_lower)

        removed = sorted(set(original_words) - set(simplified_words))
        added = sorted(set(simplified_words) - set(original_words))

        limit = min(len(removed), len(added))
        for i in range(limit):
            changes.append((removed[i], added[i]))

        return changes

    def build_credibility_reason(self, label, confidence, prob_map):
        sorted_scores = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        top_scores = sorted_scores[:2]

        parts = []
        for name, score in top_scores:
            parts.append(f"{name}: {score:.2f}")

        score_text = ", ".join(parts)

        return f"Predicted '{label}' with confidence {confidence:.2f}. Top scores: {score_text}."

    def run(self, text):
        text = self.clean_text(text)
        simplified = self.simplify_text(text)
        cred_label, cred_confidence, cred_probs = self.classify_credibility(text)
        term_changes = self.extract_term_changes(text, simplified)
        cred_reason = self.build_credibility_reason(cred_label, cred_confidence, cred_probs)

        return {
            "original": text,
            "simplified": simplified,
            "credibility_label": cred_label,
            "confidence": round(cred_confidence, 4),
            "credibility_reason": cred_reason,
            "credibility_scores": {k: round(v, 4) for k, v in cred_probs.items()},
            "term_explanations": term_changes,
        }


def main():
    pipeline = MediClearNeuralPipeline()

    print("MediClear Neural Pipeline")
    print("Type medical text and press enter.")
    print("Type 'quit' to stop.\n")

    while True:
        text = input("Input: ").strip()

        if text.lower() == "quit":
            break

        if text == "":
            print("Please enter some text.\n")
            continue

        result = pipeline.run(text)

        print("\n--- RESULT ---")
        print("Original:", result["original"])
        print("Simplified:", result["simplified"])
        print("Credibility Label:", result["credibility_label"])
        print("Confidence:", result["confidence"])
        print("Reason:", result["credibility_reason"])
        print("Scores:", result["credibility_scores"])

        if len(result["term_explanations"]) > 0:
            print("Term Changes:")
            for old_word, new_word in result["term_explanations"]:
                print(f"  - {old_word} -> {new_word}")

        print()


if __name__ == "__main__":
    main()
