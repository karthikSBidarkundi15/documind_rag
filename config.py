import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Paths ---
DATA_DIR = "data/raw"
VECTORSTORE_DIR = "vectorstore"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4

# --- LLM ---
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2