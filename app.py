import streamlit as st
from grading import grade_report
from extract import extract_text_from_pdf

st.title("GradeAI")

# Dropdown to select LLM
model_options = {
    "Meta-Llama 3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Gemma": "gemma/gemma-llm-model",
    "Deepseek 7B Chat": "deepseek-ai/deepseek-llm-7b-chat"
}

selected_model = st.selectbox("Select LLM to grade report", list(model_options.keys()))
model_name = model_options[selected_model]

# Upload PDF
uploaded_file = st.file_uploader("Upload your report (PDF only)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if st.button("Grade Report"):
        with st.spinner("Report is being graded..."):
            result = grade_report(text, model_name=model_name)

        if isinstance(result, dict) and "feedback" in result:
            st.subheader("Result")
            st.json(result)
        else:
            st.error("Failed to parse. Raw output:")
            st.text(result.get("raw_response") if isinstance(result, dict) else str(result))
