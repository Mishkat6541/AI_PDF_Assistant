import streamlit as st
from pdf_qa import process_pdf, ask_question

st.set_page_config(page_title="PDF Research Assistant", layout="centered")

st.title("📄🧠 PDF Research Assistant")
st.markdown("Upload a PDF and ask questions like you're chatting with it!")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        db = process_pdf(uploaded_file)

    st.success("PDF is ready! Ask your question.")

    query = st.text_input("Enter your question about the PDF:")

    if query and db:
        with st.spinner("Thinking..."):
            response = ask_question(query, db)
            st.write("💬 Answer:")
            st.markdown(response)
