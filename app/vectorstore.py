import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ---------- FIXED FOR STREAMLIT CLOUD ----------
# Use one stable in-memory collection name
COLLECTION_NAME = "rag_system_collection"

def _embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )


# ---------- BUILD VECTORSTORE ----------
def build_vectorstore(docs, persist: bool = False):
    """
    Cloud-safe Chroma store (in-memory only).
    """
    embed = _embeddings()

    store = Chroma.from_documents(
        documents=docs,
        embedding=embed,
        persist_directory=None,       # MUST be None on Streamlit Cloud
        collection_name=COLLECTION_NAME,
    )
    return store


# ---------- LOAD VECTORSTORE ----------
def load_vectorstore():
    """
    Returns an empty in-memory vectorstore if nothing exists.
    Cloud resets between runs, so persistence is not supported.
    """
    embed = _embeddings()

    try:
        store = Chroma(
            persist_directory=None,
            collection_name=COLLECTION_NAME,
            embedding_function=embed,
        )
        return store
    except:
        return None


# ---------- CLEAR VECTORSTORE ----------
def clear_chroma():
    """
    Nothing to delete since persistence is disabled.
    This function simply returns True to keep UI stable.
    """
    return True