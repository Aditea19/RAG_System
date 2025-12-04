import os
import shutil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from app.config import CHROMA_PERSIST_DIR


def _embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ----------------------------
# BUILD VECTORSTORE (In-Memory)
# ----------------------------
def build_vectorstore(docs, persist: bool = True):
    embed = _embeddings()

    store = InMemoryVectorStore.from_documents(
        docs,
        embedding=embed
    )

    # Optional: Save text + embeddings locally (only works locally, ignored on cloud)
    if persist:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        store.save_local(CHROMA_PERSIST_DIR)

    return store


# ----------------------------
# LOAD VECTORSTORE
# ----------------------------
def load_vectorstore():
    embed = _embeddings()

    try:
        return InMemoryVectorStore.load_local(
            CHROMA_PERSIST_DIR,
            embeddings=embed
        )
    except:
        return None


# ----------------------------
# CLEAR VECTORSTORE
# ----------------------------
def clear_chroma():
    try:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        time.sleep(0.3)
        return True
    except:
        return False