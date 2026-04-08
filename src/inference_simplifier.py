from pathlib import Path
import re
from difflib import SequenceMatcher

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_DIR = Path("models/simplifier")
MODEL_NAME = "t5-small"
PREFIX = "Simplify this medical text into plain English: "
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 64
MIN_OUTPUT_LENGTH = 5
OUTPUT_PREFIX_PATTERNS = [
    r"^in this review[,:\s]+",
    r"^simplify this medical text into plain english[:\s]+",
    r"^simplify this medical text into simple english[:\s]+",
    r"^simplify this medical text into simple terms[:\s]+",
    r"^we found that\s+",
    r"^we found\s+",
    r"^this review found that\s+",
    r"^this review found\s+",
    r"^the review found that\s+",
    r"^the review found\s+",
    r"^review authors found that\s+",
    r"^review authors found\s+",
    r"^this review included\s+",
    r"^this review aims to [^.]+\.\s*",
    r"^this review includes [^.]+\.\s*",
    r"^the findings of this review (were|are)\s+",
]
BAD_PHRASES = [
    "in this review",
    "we found",
    "this review",
    "systematic review",
    "simplify this medical text",
]
BAD_JARGON_TERMS = [
    "diabetes mellitus",
    "blood glucose",
    "contraindicated",
    "hepatic impairment",
    "prognosis",
    "adherence to therapy",
]
TERM_REPLACEMENTS = [
    (r"\bhypertension\b", "high blood pressure"),
    (r"\bsodium intake\b", "salt intake"),
    (r"\bmyocardial infarction\b", "heart attack"),
    (r"\bmetastatic disease\b", "cancer that has spread"),
    (r"\bedema\b", "swelling"),
    (r"\bcardiac dysfunction\b", "heart problem"),
    (r"\borally\b", "by mouth"),
    (r"\btwice daily\b", "two times a day"),
    (r"\bdiabetes mellitus\b", "diabetes"),
    (r"\bmonitor blood glucose\b", "check blood sugar"),
    (r"\bblood glucose\b", "blood sugar"),
    (r"\bdyspnea\b", "shortness of breath"),
    (r"\bmild exertion\b", "light activity"),
    (r"\btopical corticosteroids\b", "steroid creams"),
    (r"\binflammation\b", "swelling"),
    (r"\bpruritus\b", "itching"),
    (r"\bhistopathological examination\b", "lab testing"),
    (r"\bbenign\b", "not cancer"),
    (r"\brenal complications\b", "kidney problems"),
    (r"\brenal insufficiency\b", "kidney problems"),
    (r"\bis contraindicated in\b", "should not be used in"),
    (r"\bhepatic impairment\b", "serious liver problems"),
    (r"\bsevere serious liver problems\b", "serious liver problems"),
    (r"\bcontraindicated\b", "should not be used"),
    (r"\bthromboembolic risk\b", "risk of blood clots"),
    (r"\bchronic obstructive pulmonary disease\b", "COPD"),
    (r"\bgas exchange\b", "oxygen exchange"),
    (r"\bthe prognosis is favorable\b", "recovery is likely to go well"),
    (r"\badherence to therapy\b", "following treatment"),
    (r"\bexacerbate\b", "get worse"),
    (r"\bnausea\b", "feeling sick"),
    (r"\bfatigue\b", "tiredness"),
]
PHRASE_REPLACEMENTS = [
    (r"\bpatients\b", "people"),
    (r"\bindividuals\b", "people"),
    (r"\badminister the medication\b", "take the medicine"),
    (r"\bmaintain adequate hydration\b", "drink enough water"),
    (r"\bschedule a follow-up consultation\b", "make a follow-up appointment"),
    (r"\bdiscontinue use immediately\b", "stop using it right away"),
    (r"\bundergoing chemotherapy\b", "getting chemotherapy"),
    (r"\bis experiencing\b", "has"),
    (r"\bmonitor\b", "check"),
    (r"\bpostoperatively\b", "after surgery"),
    (r"\bambulation\b", "walking"),
]


def load_model_and_tokenizer(model_dir: Path = MODEL_DIR):
    source = model_dir if model_dir.exists() and any(model_dir.iterdir()) else Path(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(str(source))
    model = T5ForConditionalGeneration.from_pretrained(str(source))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def postprocess_output(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    for pattern in OUTPUT_PREFIX_PATTERNS:
        updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        if updated != cleaned:
            cleaned = updated.strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()


def fallback_simplify(text: str) -> str:
    simplified = f" {text.strip()} "
    for pattern, replacement in TERM_REPLACEMENTS + PHRASE_REPLACEMENTS:
        simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

    simplified = " ".join(simplified.strip().split())
    simplified = re.sub(r"\bshould monitor\b", "should check", simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"\bshould reduce\b", "should reduce", simplified, flags=re.IGNORECASE)

    if simplified and simplified[0].islower():
        simplified = simplified[0].upper() + simplified[1:]
    return simplified


def should_use_fallback(source_text: str, generated_text: str) -> bool:
    lowered = generated_text.lower()
    if any(phrase in lowered for phrase in BAD_PHRASES):
        return True
    if any(term in lowered for term in BAD_JARGON_TERMS):
        return True
    if similarity(source_text, generated_text) >= 0.92:
        return True
    if len(generated_text) > max(len(source_text) * 1.25, len(source_text) + 20):
        return True
    return False


def simplify(text: str) -> str:
    tokenizer, model, device = load_model_and_tokenizer()
    encoded = tokenizer(
        PREFIX + text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generated = model.generate(
        **encoded,
        num_beams=4,
        min_length=MIN_OUTPUT_LENGTH,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=0.9,
        max_length=MAX_OUTPUT_LENGTH,
        early_stopping=True,
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    cleaned = postprocess_output(decoded)
    if should_use_fallback(text, cleaned):
        return postprocess_output(fallback_simplify(text))
    return cleaned


if __name__ == "__main__":
    examples = [
        "Patients with hypertension should reduce sodium intake.",
        "Myocardial infarction requires immediate medical attention.",
        "Metastatic disease indicates the cancer has spread.",
    ]

    for ex in examples:
        print("Input:", ex)
        print("Output:", simplify(ex))
        print()
