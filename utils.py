from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(pdf_path):
    """
    Load PDF and return LangChain documents.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def create_chunks(documents):
    """
    Split documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    return chunks