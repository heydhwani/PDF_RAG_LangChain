import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# =======================
# 🔑 Load API Key
# =======================
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# =======================
# 🎨 UI
# =======================
st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("📄 AI PDF Chatbot")
st.markdown("### 🤖 Ask questions from any PDF")

st.info("👉 Upload a PDF and ask questions related to its content.")

# =======================
# 📄 Upload
# =======================
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()

    if not documents:
        st.error("❌ Unable to read PDF")
        st.stop()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=gemini_key
    )

    # =======================
    # 💡 Example Questions
    # =======================
    st.subheader("💡 Try asking:")
    example_qs = [
        "What is this document about?",
        "Summarize this PDF",
        "Explain the main concept",
        "List key points",
        "What are the advantages mentioned?"
    ]

    cols = st.columns(len(example_qs))
    for i, q in enumerate(example_qs):
        if cols[i].button(q):
            st.session_state.query = q

    # =======================
    # 🔍 Input
    # =======================
    query = st.text_input(
        "🔍 Ask your question:",
        value=st.session_state.get("query", "")
    )

    # =======================
    # 💬 Answer
    # =======================
    if query:
        with st.spinner("⏳ Thinking..."):
            docs = db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer based on this context:
            {context}

            Question: {query}
            """

            response = llm.invoke(prompt)

        st.subheader("💡 Answer")
        st.write(response.content)

        # Optional: sources
        with st.expander("📄 See extracted content"):
            for i, d in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(d.page_content[:300])
                st.write("---")

else:
    st.warning("⚠️ Please upload a PDF to start")