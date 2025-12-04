from pathlib import Path
import os
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# LLM
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


# Stored in session so every rerun gets correct name
if "CHROMA_COLLECTION_NAME" not in globals():
    CHROMA_COLLECTION_NAME = f"enterprise_rag_{uuid.uuid4().hex[:8]}"

def reset_chroma_collection_name():
    """Generate a fresh unique collection name."""
    global CHROMA_COLLECTION_NAME
    CHROMA_COLLECTION_NAME = f"enterprise_rag_{uuid.uuid4().hex[:8]}"

def get_chroma_collection_name():
    """Always return the latest collection name."""
    return CHROMA_COLLECTION_NAME


CHROMA_PERSIST_DIR = str(VECTOR_DIR / "chroma_db")