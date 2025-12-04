import shutil
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Embeddings: use Google, not HuggingFace (HuggingFace requires PyTorch)
def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )


# Build vectorstore (fully in-memory)
def build_vectorstore(docs, persist: bool = False):
    embed = _embeddings()

    store = InMemoryVectorStore.from_documents(
        docs,
        embedding=embed
    )

    return store


# Load vectorstore — in-memory vectorstore cannot be loaded
def load_vectorstore():
    return None   # Always start fresh


# Clear chroma directory (not used anymore)
def clear_chroma():
    try:
        shutil.rmtree("vectorstore", ignore_errors=True)
        time.sleep(0.2)
        return True
    except:
        return False