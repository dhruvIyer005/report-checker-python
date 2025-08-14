import streamlit as st
from grading import grade_report
from extract import extract_text_from_pdf

st.title("Llama Report grader")
uploaded_file=st.file_uploader("upload your report(IN PDF ONLY)",type="pdf")

if uploaded_file:
    text=extract_text_from_pdf(uploaded_file)
    if st.button("Grade Report"):
        with st.spinner("Report is being graded.."):
            result=grade_report(text)


        if isinstance(result,dict):
            st.subheader("Result")
            st.json(result)

        else:
            st.error("Failed to parse, Raw output:")
            st.text(result) 