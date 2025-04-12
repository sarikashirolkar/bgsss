import streamlit as st
from file_loader import load_pdf_text
from embedder import embed_and_index
from qa_chain import ask_question
import tempfile
import os

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📘 Ask Your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully. Processing...")

    # Load, embed, and index
    text = load_pdf_text(pdf_path)
    chunks, index, embed_model = embed_and_index(text)

    st.success("✅ PDF processed. You can ask your question now.")

    question = st.text_input("Ask a question based on the PDF")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = ask_question(question, chunks, index, embed_model)
            st.markdown(f"### 🧠 Answer:\n{answer}")
