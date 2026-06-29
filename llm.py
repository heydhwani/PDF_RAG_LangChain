from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL
)


def load_gemini():

    if not GEMINI_API_KEY:
        return None

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY
    )


def load_groq():

    if not GROQ_API_KEY:
        return None

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )