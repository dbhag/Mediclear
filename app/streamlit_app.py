import streamlit as st
from mediclear.neural_pipeline import MediClearNeuralPipeline


st.set_page_config(
    page_title="MediClear Neural",
    page_icon="🩺",
    layout="centered"
)


@st.cache_resource
def load_pipeline():
    return MediClearNeuralPipeline()


examples = {
    "": "",
    "High blood pressure advice": "Patients with hypertension should reduce sodium intake.",
    "Emergency guidance": "Patients experiencing myocardial infarction symptoms should seek immediate intervention.",
    "Suspicious claim": "Drinking lemon water cures cancer instantly.",
    "Diabetes advice": "Patients with diabetes should monitor their blood glucose levels regularly.",
    "Asthma guidance": "Individuals with asthma should avoid triggers such as smoke and pollen.",
    "Flu prevention": "Getting vaccinated can reduce the risk of influenza infection.",
    "Medication reminder": "Patients should take antibiotics exactly as prescribed by their physician.",
    "Cancer explanation": "Metastatic disease means that the cancer has spread to other parts of the body.",
    "Cholesterol advice": "High cholesterol can increase the risk of cardiovascular disease.",
    "Stroke warning": "Sudden weakness on one side of the body may be a symptom of stroke.",
    "Dehydration advice": "Severe dehydration may require immediate medical treatment.",
    "Exercise benefit": "Regular physical activity can improve heart health.",
    "Bad claim about vaccines": "Vaccines cause autism in children.",
    "Bad claim about salt water": "Drinking salt water cures all infections instantly.",
    "Bad claim about herbs": "Herbal tea alone can completely treat pneumonia.",
    "Bad claim about diabetes": "People with diabetes do not need medicine if they drink enough water.",
    "Neutral claim": "Some studies suggest that exercise may improve mental health.",
    "Uncertain claim": "Vitamin supplements may help some people, but evidence is mixed.",
}


st.title("MediClear Neural 🩺")
st.caption("Medical text simplification and credibility prediction")


col1, col2 = st.columns([1, 2])

with col1:
    choice = st.selectbox("Try an example:", list(examples.keys()))

with col2:
    text = st.text_area(
        "Enter medical text:",
        value=examples[choice],
        height=140
    )


pipeline = None
load_error = None

try:
    pipeline = load_pipeline()
except Exception as e:
    load_error = str(e)


if load_error:
    st.error("Could not load the pipeline.")
    st.code(load_error)


if st.button("Run MediClear", use_container_width=True):
    if load_error:
        st.error("The pipeline is not loaded, so the app cannot run.")

    elif not text.strip():
        st.warning("Please enter some text.")

    else:
        try:
            with st.spinner("Running neural pipeline..."):
                result = pipeline.run(text)

            st.subheader("Results")

            st.markdown("### Text Output")
            st.write("**Original:**")
            st.write(result["original"])

            st.write("**Simplified:**")
            st.write(result["simplified"])

            label = str(result["credibility_label"]).strip().lower()
            confidence = round(float(result["confidence"]), 3)

            st.markdown("### Credibility Summary")

            m1, m2 = st.columns(2)

            with m1:
                st.metric("Predicted Label", label.title())

            with m2:
                st.metric("Confidence", confidence)

            if label == "true":
                st.success("This claim was predicted as True.")
            elif label == "false":
                st.error("This claim was predicted as False.")
            elif label == "mixture":
                st.warning("This claim was predicted as Mixture.")
            else:
                st.info("This claim was predicted as Unproven.")

            reason = result.get("credibility_reason", "")
            if reason != "":
                st.caption(reason)

            scores = result.get("credibility_scores", {})
            if len(scores) > 0:
                st.markdown("### Class Probabilities")

                score_rows = []
                for name in scores:
                    score_rows.append({
                        "Label": name,
                        "Score": round(float(scores[name]), 4)
                    })

                st.dataframe(score_rows, use_container_width=True)

            changes = result.get("term_explanations", [])
            if len(changes) > 0:
                st.markdown("### Detected Term Changes")

                for old_word, new_word in changes:
                    st.write(f"- **{old_word}** → {new_word}")

        except Exception as e:
            st.error("Something went wrong while running the pipeline.")
            st.code(str(e))