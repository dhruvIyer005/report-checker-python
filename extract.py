import pdfplumber

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text="\n".join([page.extract_text() for page in pdf.pages])
        return text