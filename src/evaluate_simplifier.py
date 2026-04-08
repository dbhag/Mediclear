import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_simplifier import simplify


EXAMPLES = [
    "Patients with hypertension should reduce sodium intake.",
    "Myocardial infarction requires immediate medical attention.",
    "Metastatic disease indicates the cancer has spread.",
    "Persistent edema may indicate underlying cardiac dysfunction.",
    "Administer the medication orally twice daily after meals.",
    "Individuals with diabetes mellitus should monitor blood glucose regularly.",
    "Discontinue use immediately if an allergic reaction occurs.",
    "Topical corticosteroids can reduce inflammation and pruritus.",
    "The lesion was determined to be benign after histopathological examination.",
    "The patient is experiencing dyspnea during mild exertion.",
    "Maintain adequate hydration to prevent renal complications.",
    "Schedule a follow-up consultation if symptoms exacerbate.",
    "The prognosis is favorable with early intervention and adherence to therapy.",
    "Patients undergoing chemotherapy may experience nausea and fatigue.",
    "The medication is contraindicated in individuals with severe hepatic impairment.",
    "Ambulation is encouraged postoperatively to reduce thromboembolic risk.",
    "Chronic obstructive pulmonary disease can impair gas exchange.",
    "Renal insufficiency may require dosage adjustment.",
]
BAD_PHRASES = ["in this review", "we found", "this review", "systematic review"]
OUTPUT_FILE = ROOT / "simplifier_examples.md"


def contains_bad_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in BAD_PHRASES)


def main() -> None:
    lines = ["# Simplifier Examples", ""]

    for index, text in enumerate(EXAMPLES, start=1):
        output = simplify(text)
        identical = output.strip() == text.strip()
        bad_phrase = contains_bad_phrase(output)

        print(f"Example {index}")
        print("Input:", text)
        print("Output:", output)
        print("Identical:", identical)
        print("Contains bad phrase:", bad_phrase)
        print("Length ratio:", f"{len(output)}/{len(text)}")
        if identical:
            print("Warning: output is extremely close to the input.")
        print()

        lines.append(f"## Example {index}")
        lines.append("")
        lines.append(f"- Input: {text}")
        lines.append(f"- Output: {output}")
        lines.append(f"- Identical: {identical}")
        lines.append(f"- Contains bad phrase: {bad_phrase}")
        lines.append(f"- Length ratio: {len(output)}/{len(text)}")
        lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
