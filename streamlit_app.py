import os
import streamlit as st

from utils import load_pdf, create_chunks
from vector_store import create_vector_store
from llm import load_gemini, load_groq
from prompts import get_prompt


# -------------------- Configuration --------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    layout="wide"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------- UI --------------------

st.title("📄 AI PDF Chatbot")
st.markdown("### Ask Questions from Any PDF")

st.info("Upload a PDF and ask questions related to its content.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# -------------------- Main --------------------

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    try:

        # Load PDF
        documents = load_pdf(pdf_path)

        if not documents:
            st.error("Unable to read PDF.")
            st.stop()

        # Create chunks
        chunks = create_chunks(documents)

        # Build Vector Store
        db = create_vector_store(chunks)

        # Load LLMs
        gemini = load_gemini()
        groq = load_groq()

        # ---------------- Example Questions ----------------

        st.subheader("💡 Try Asking")

        example_questions = [
            "What is this document about?",
            "Summarize this PDF",
            "Explain the main concept",
            "List key points",
            "What are the advantages mentioned?"
        ]

        cols = st.columns(len(example_questions))

        for i, question in enumerate(example_questions):
            if cols[i].button(question):
                st.session_state["query"] = question

        # ---------------- User Query ----------------

        query = st.text_input(
            "Ask your question",
            value=st.session_state.get("query", "")
        )

        if query:

            with st.spinner("Searching relevant content and generating response..."):

                # ---------------- Summarization ----------------

                if any(
                    word in query.lower()
                    for word in ["summarize", "summary", "overview"]
                ):

                    context = "\n\n".join(
                        doc.page_content
                        for doc in documents
                    )

                # ---------------- RAG ----------------

                else:

                    docs = db.similarity_search(
                        query=query,
                        k=5
                    )

                    if not docs:
                        st.warning("No relevant information found.")
                        st.stop()

                    context = "\n\n".join(
                        doc.page_content.strip()
                        for doc in docs
                    )

                # ---------------- Prompt ----------------

                prompt = get_prompt(
                    context=context,
                    question=query
                )

                # ---------------- LLM ----------------

                try:

                    if gemini is not None:
                        response = gemini.invoke(prompt)
                    else:
                        raise Exception("Gemini unavailable")

                except Exception as e:

                    if (
                        "RESOURCE_EXHAUSTED" in str(e)
                        or "429" in str(e)
                        or "quota" in str(e).lower()
                    ):

                        if groq is None:
                            st.error(
                                "AI service is temporarily unavailable."
                            )
                            st.stop()

                        response = groq.invoke(prompt)

                    else:
                        raise e

            st.markdown("## 🤖 AI Assistant")
            st.write(response.content)

    except Exception as e:
        st.error(f"❌ {e}")

else:
    st.warning("Please upload a PDF to start.")