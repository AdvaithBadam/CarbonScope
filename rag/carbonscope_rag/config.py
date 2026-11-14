"""Central settings for the RAG module. Change values here, not inside other files."""
from pathlib import Path

from dotenv import load_dotenv

# rag/ folder and the repo root, resolved from this file so paths work from anywhere
RAG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = RAG_ROOT.parent

# .env at repo root or inside rag/ (already-set environment variables win)
load_dotenv(REPO_ROOT / ".env")
load_dotenv(RAG_ROOT / ".env")

# --- Data locations ---
DATA_DIR = RAG_ROOT / "data" / "pdf"
VECTOR_DIR = RAG_ROOT / "data" / "vector_store"   # gitignored, rebuilt by scripts/ingest.py
COLLECTION_NAME = "ccts_docs"

# --- Chunking / embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5
# Cosine similarity cutoff. Tune it: ask a few off-topic questions and a few
# on-topic ones with scripts/ask.py --show-context and pick a value between them.
MIN_SCORE = 0.30

# --- LLM (key is read from the GROQ_API_KEY environment variable) ---
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
MAX_HISTORY_MESSAGES = 6   # how many previous chat messages are sent to the LLM
