# 📄 AI PDF Chatbot using RAG, LangChain, Gemini & Groq

An AI-powered PDF Question Answering application built using **Retrieval-Augmented Generation (RAG)**. Upload any PDF and ask natural language questions about its content. The application retrieves the most relevant document chunks using **FAISS** and generates context-aware responses using **Google Gemini**, with **Groq** as an automatic fallback when Gemini is unavailable or its quota is exhausted.

---

## 🚀 Live Demo

🔗 **Live App:** *https://pdfraglangchaingit-3zkimyfvyxqtgvvdbhjlmx.streamlit.app/*

---

## 📌 Features

- 📄 Upload any PDF document
- ✂️ Automatic document chunking
- 🔍 Semantic similarity search using FAISS
- 🤖 AI-powered question answering
- 📝 Full document summarization
- ⚡ Google Gemini as the primary LLM
- 🔁 Automatic Groq fallback
- 💡 Example question buttons
- 🎯 Context-grounded responses to reduce hallucinations

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Framework | LangChain |
| LLMs | Google Gemini, Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| Language | Python |

---

## 📂 Project Structure

```text
PDF_RAG_LANGCHAIN/
│
├── docs/
│   └── rag_paper.pdf
│
├── uploads/
├── vector_store/
│
├── streamlit_app.py
├── utils.py
├── vector_store.py
├── llm.py
├── prompts.py
├── config.py
├── requirements.txt
├── README.md
└── .env
```

---

## ⚙️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root.


### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

---

## 🔄 RAG Workflow

1. Upload a PDF document.
2. Extract text using **PyMuPDF**.
3. Split the document into overlapping chunks.
4. Generate embeddings using **HuggingFace Sentence Transformers**.
5. Store embeddings in **FAISS**.
6. Retrieve the most relevant chunks for the user query.
7. Create a context-aware prompt.
8. Generate a response using **Google Gemini**.
9. Automatically switch to **Groq** if Gemini is unavailable.

---

## 💬 Example Questions

- What is this document about?
- Summarize this PDF
- Explain the main concept
- List key points
- What are the advantages mentioned?

---

## 🔮 Future Improvements

- Multi-PDF support
- Chat history
- Source citations for answers
- Hybrid Search (BM25 + Vector Search)
- Streaming responses
- Conversation memory

---

## 👩‍💻 Author

**Dhwani Jain**