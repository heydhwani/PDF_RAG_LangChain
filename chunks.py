from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Same embedding model used while creating the index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the saved FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",  # Replace with your folder name
    embeddings,
    allow_dangerous_deserialization=True
)

# Total vectors (chunks)
print(f"Total chunks in vector DB: {vectorstore.index.ntotal}")

# Total documents in docstore
print(f"Total documents: {len(vectorstore.docstore._dict)}")