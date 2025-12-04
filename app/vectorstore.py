import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil, time
from app.config import get_chroma_collection_name, CHROMA_PERSIST_DIR


def build_vectorstore(docs, persist: bool = True):
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # On Streamlit Cloud: ALWAYS use in-memory (no persistence)
    if os.environ.get("STREAMLIT_CLOUD_APP") or "streamlit.io" in os.environ.get("HOME", ""):
        persist_dir = None
    else:
        persist_dir = CHROMA_PERSIST_DIR if persist else None
    
    store = Chroma.from_documents(
        documents=docs, 
        embedding=embed, 
        persist_directory=persist_dir, 
        collection_name=get_chroma_collection_name()
    )
    return store


def load_vectorstore():
    try:
        embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # On Streamlit Cloud: ALWAYS use in-memory (no persistence)
        if os.environ.get("STREAMLIT_CLOUD_APP") or "streamlit.io" in os.environ.get("HOME", ""):
            persist_dir = None
        else:
            persist_dir = CHROMA_PERSIST_DIR
        
        store = Chroma(
            persist_directory=persist_dir, 
            collection_name=get_chroma_collection_name(), 
            embedding_function=embed
        )
        return store
    except:
        return None


# Chroma clearing
def clear_chroma():
    try:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        time.sleep(0.4)  # allow windows to release locks
        return True
    except:
        return False
