import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# UI Title
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot")
st.write("Ask questions based on the uploaded PDF")

# Sidebar - Example Questions
st.sidebar.header("💡 Example Questions")
example_questions = [
    "What is RAG?",
    "Explain RAG architecture",
    "Difference between RAG-Sequence and RAG-Token?",
    "How does RAG work?",
    "What is DPR?",
    "Applications of RAG?"
]

for q in example_questions:
    if st.sidebar.button(q):
        st.session_state.query = q

# Load PDF
pdf_path = "docs/rag_paper.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=gemini_key
)

# Input box
query = st.text_input("🔍 Ask your question:")

if query:
    # Retrieve
    results = vectorstore.similarity_search(query, k=3)

    # Show context (optional)
    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(results):
            st.write(f"Chunk {i+1}:")
            st.write(doc.page_content)
            st.write("---")

    # Create prompt
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {query}

    Answer clearly in 3-4 sentences.
    """

    # Generate response
    response = llm.invoke(prompt)

    # Display
    st.subheader("💡 Answer")
    st.write(response.content)