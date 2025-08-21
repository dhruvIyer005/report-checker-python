import streamlit as st
from grading import grade_report
from extract import extract_text_from_pdf

st.title("📘 GradeAI - Report Grader")

# Dropdown to select LLM
model_options = {
    "Meta-LLaMA 3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Gemma 2 9B Instruct": "google/gemma-2-9b-it",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.2"
}


selected_model = st.selectbox("Select LLM to grade report", list(model_options.keys()))
model_name = model_options[selected_model]

# Upload PDF
uploaded_file = st.file_uploader("Upload your report (PDF only)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if st.button("Grade Report"):
        with st.spinner("⏳ Report is being graded..."):
            result = grade_report(text, model_name=model_name)

        if isinstance(result, dict) and "feedback" in result:
            st.subheader("✅ Result")
            st.json(result)
        else:
            st.error("⚠️ Failed to parse JSON from model response.")
            st.subheader("Raw Output")
            st.text(result.get("raw_response") if isinstance(result, dict) else str(result))
