import os
from dotenv import load_dotenv

load_dotenv()

# API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash-lite"

# Chunk Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector Store
VECTOR_DB_PATH = "vector_store"