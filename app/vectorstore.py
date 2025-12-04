import os
import shutil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import get_chroma_collection_name, CHROMA_PERSIST_DIR


def _embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ----------------------------
# BUILD VECTORSTORE (FAISS)
# ----------------------------
def build_vectorstore(docs, persist: bool = True):
    embed = _embeddings()

    store = FAISS.from_documents(docs, embed)

    # Local persistence (Streamlit Cloud ignores this safely)
    if persist:
        store.save_local(CHROMA_PERSIST_DIR)

    return store


# ----------------------------
# LOAD VECTORSTORE
# ----------------------------
def load_vectorstore():
    embed = _embeddings()

    try:
        return FAISS.load_local(
            CHROMA_PERSIST_DIR,
            embed,
            allow_dangerous_deserialization=True  # required for FAISS
        )
    except:
        return None


# ----------------------------
# CLEAR VECTORSTORE
# ----------------------------
def clear_chroma():  # keep same function name for compatibility
    try:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        time.sleep(0.3)
        return True
    except:
        return False