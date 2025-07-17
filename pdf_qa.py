import fitz  # PyMuPDF
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os
from tempfile import NamedTemporaryFile

# Set your OpenAI key here or use environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(file):
    raw_text = extract_text_from_pdf(file)

    # Split into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)

    # Create Document objects
    docs = [Document(page_content=t) for t in texts]

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

def ask_question(query, vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    return result
