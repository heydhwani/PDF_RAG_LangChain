import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, VECTOR_DB_PATH

# Load embedding model only once
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def create_vector_store(chunks):
    """
    Create FAISS vector store and save it locally.
    """

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(VECTOR_DB_PATH)

    return db


def load_vector_store():
    """
    Load existing vector store.
    """

    if not os.path.exists(VECTOR_DB_PATH):
        return None

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


def get_vector_store(chunks=None):
    """
    Returns an existing vector store if available,
    otherwise creates a new one.
    """

    db = load_vector_store()

    if db is not None:
        return db

    if chunks is None:
        raise ValueError("Chunks are required to build vector store.")

    return create_vector_store(chunks)