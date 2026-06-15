# 📄 AI PDF Chatbot — RAG Pipeline with LangChain & Gemini


A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask natural language questions. The system retrieves the most relevant chunks from the document and generates accurate, context-grounded answers using Google's Gemini LLM — without hallucinating outside the document.
*Test App here*
https://pdfraglangchaingit-3zkimyfvyxqtgvvdbhjlmx.streamlit.app/


---


## 🧠 How It Works

```
PDF File
   ↓
PyPDFLoader (extract text)
   ↓
RecursiveCharacterTextSplitter (chunk: 500 tokens, overlap: 100)
   ↓
HuggingFace Embeddings — sentence-transformers/all-MiniLM-L6-v2
   ↓
FAISS Vector Store (similarity search)
   ↓
Top-3 relevant chunks retrieved
   ↓
Custom Prompt + Gemini 2.0 Flash Lite (LLM)
   ↓
Context-grounded Answer
```

---

## 🛠️ Tech Stack

| Component | Tool Used |
|---|---|
| PDF Loader | `PyPDFLoader` (LangChain) |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | `FAISS` |
| LLM | `Gemini 2.0 Flash Lite` (Google Generative AI) |
| Framework | `LangChain` |
| Env Management | `python-dotenv` |

---

## 📁 Project Structure

```
PDF_RAG_LangChain/
│
├── docs/
│   └── rag_paper.pdf        # Sample PDF for testing
│
├── app.py                   # Main RAG pipeline
├── requirements.txt         # Dependencies
├── .env                     # API keys (not committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/heydhwani/PDF_RAG_LangChain.git
cd PDF_RAG_LangChain
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Key
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
Get your free Gemini API key from: https://aistudio.google.com/

### 5. Add Your PDF
Place your PDF inside the `docs/` folder and update the path in `app.py`:
```python
pdf_path = "docs/your_file.pdf"
```

### 6. Run the App
```bash
python app.py
```

---

## 💡 Key Features

- **Strict document grounding** — LLM answers ONLY from the PDF, no hallucination
- **Semantic search** — FAISS finds the most relevant chunks using vector similarity
- **HuggingFace embeddings** — No paid embedding API needed
- **Custom prompt engineering** — Structured prompt ensures precise, formatted answers
- **Resume-aware output** — Formats answers in bullet points when document is a resume

---

## 📌 Example

```
Ask a question: What methodology does the paper use?

Final Answer:
The paper proposes a RAG (Retrieval-Augmented Generation) approach that combines:
- A dense retriever for fetching relevant document passages
- A sequence-to-sequence model for generating the final answer
- Non-parametric memory via a document index
```

---

## 📦 Requirements

```
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-google-genai
faiss-cpu
pypdf
sentence-transformers
python-dotenv
```

---

## 🔮 Future Improvements

- [ ] Add Streamlit UI for browser-based interaction
- [ ] Support multiple PDF uploads
- [ ] Add chat history / multi-turn conversation
- [ ] Swap FAISS with ChromaDB for persistent storage
- [ ] Deploy on Hugging Face Spaces

---

## 👩‍💻 Author

**Dhwani Jain**  
B.Tech CSE | Ajay Kumar Garg Engineering College  
[LinkedIn](https://www.linkedin.com/in/dhwani-jain-67508327a/) • [GitHub](https://github.com/heydhwani)
