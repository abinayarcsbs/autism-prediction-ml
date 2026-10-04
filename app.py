import streamlit as st
import pandas as pd
import joblib

# ---------- Load saved artifacts ----------
model_full = joblib.load('asd_model.pkl')
model_reduced = joblib.load('asd_model_reduced.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="ASD Prediction", page_icon="🧩")
st.title("🧩 Autism Spectrum Disorder (ASD) Screening")
st.write("Answer the AQ-10 questions and provide basic details to get a prediction.")

# ---------- Screening mode toggle ----------
st.subheader("Choose Screening Mode")
mode = st.radio(
    "Screening Mode",
    ["Full Screen (10 questions) — 95.7% accuracy", "Quick Screen (5 questions) — 85.0% accuracy"],
    label_visibility="collapsed"
)
is_quick = mode.startswith("Quick")

st.caption(
    "🔬 **Research note:** We compared a shortened 5-question version against the full "
    "AQ-10. The full version is more accurate (95.7% vs 85.0%), showing each question "
    "carries meaningful diagnostic signal — shortening the test is a real trade-off, not a free win."
)

# ---------- AQ-10 Questions ----------
all_questions = {
    "A1_Score": "I often notice small sounds when others do not.",
    "A2_Score": "I usually concentrate more on the whole picture, rather than the small details.",
    "A3_Score": "I find it easy to do more than one thing at once.",
    "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly.",
    "A5_Score": "I find it easy to 're-read between the lines' when someone is talking to me.",
    "A6_Score": "I know how to tell if someone listening to me is getting bored.",
    "A7_Score": "When I'm reading a story, I find it difficult to work out the characters' intentions.",
    "A8_Score": "I like to collect information about categories of things.",
    "A9_Score": "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "A10_Score": "I find it difficult to work out people's intentions."
}

top5_questions = ["A5_Score", "A9_Score", "A10_Score", "A6_Score", "A8_Score"]
questions_to_show = top5_questions if is_quick else list(all_questions.keys())

st.subheader(f"Screening Questions ({len(questions_to_show)} questions)")
scores = {}
for i, key in enumerate(questions_to_show, start=1):
    ans = st.radio(f"Q{i}. {all_questions[key]}", ["Agree", "Disagree"], key=key, horizontal=True)
    scores[key] = 1 if ans == "Agree" else 0

# ---------- Demographic inputs (only needed for Full Screen) ----------
if not is_quick:
    st.subheader("Demographic & Background Info")

    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["f", "m"])
    jundice = st.selectbox("Had jaundice at birth?", ["yes", "no"])
    austim = st.selectbox("Family member with autism?", ["yes", "no"])

    def extract_categories(prefix):
        return [c.replace(prefix, "") for c in model_columns if c.startswith(prefix)]

    ethnicity_options = extract_categories("ethnicity_")
    relation_options = extract_categories("relation_")
    country_options = extract_categories("contry_of_res_")

    ethnicity = st.selectbox("Ethnicity", ethnicity_options if ethnicity_options else ["Unknown"])
    relation = st.selectbox("Who is completing the test?", relation_options if relation_options else ["Self"])
    country = st.selectbox("Country of Residence", country_options if country_options else ["Other"])

# ---------- Predict ----------
if st.button("Predict"):
    if is_quick:
        # Reduced model — only needs the 5 A-scores
        input_df = pd.DataFrame([scores])[top5_questions]
        model = model_reduced
    else:
        # Full model — needs everything
        input_dict = dict(scores)
        input_dict["age"] = age
        input_dict["gender"] = 0 if gender == "f" else 1
        input_dict["jundice"] = 0 if jundice == "no" else 1
        input_dict["austim"] = 0 if austim == "no" else 1

        for col in model_columns:
            if col.startswith("ethnicity_") or col.startswith("relation_") or col.startswith("contry_of_res_"):
                input_dict[col] = 0

        eth_col = f"ethnicity_{ethnicity}"
        rel_col = f"relation_{relation}"
        country_col = f"contry_of_res_{country}"
        if eth_col in input_dict:
            input_dict[eth_col] = 1
        if rel_col in input_dict:
            input_dict[rel_col] = 1
        if country_col in input_dict:
            input_dict[country_col] = 1

        input_df = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)
        model = model_full

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ The model predicts a **higher likelihood of ASD traits**.")
    else:
        st.success("✅ The model predicts a **lower likelihood of ASD traits**.")

    if proba is not None:
        st.write(f"Model confidence (probability of ASD): **{proba:.2%}**")

    st.caption("This is a screening tool based on a machine learning model trained on the AQ-10 dataset. "
               "It is not a medical diagnosis. Please consult a qualified professional for an accurate assessment.")