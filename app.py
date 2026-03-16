from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

pdf_path = "docs/rag_paper.pdf"


loader = PyPDFLoader(pdf_path)


documents = loader.load()

print("Total pages:", len(documents))

print("\nFirst page content:\n")
print(documents[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0].page_content)

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert chunks into vectors
chunk_vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print("\nEmbedding dimension:", len(chunk_vectors[0]))
print("Total embeddings created:", len(chunk_vectors))

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("\nFAISS vector database created.")

# Ask user question
query = input("\nAsk a question: ")

# Search similar chunks
results = vectorstore.similarity_search(query, k=3)

print("\nTop Relevant Chunks:\n")

for i, doc in enumerate(results):
    print(f"Result {i+1}:\n")
    print(doc.page_content)
    print("\n-----------------\n")

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Explain clearly in 3-4 sentences.
"""

response = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(response.content)