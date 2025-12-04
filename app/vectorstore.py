import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import shutil, time
from app.config import get_chroma_collection_name, CHROMA_PERSIST_DIR


def build_vectorstore(docs, persist: bool = True):
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Force EphemeralClient on Streamlit Cloud (fixes tenant errors)
    if os.environ.get("STREAMLIT_CLOUD_APP") or "streamlit.io" in os.environ.get("HOME", ""):
        client = chromadb.EphemeralClient()
        persist_dir = None
    else:
        # Local: your original behavior
        persist_dir = CHROMA_PERSIST_DIR if persist else None
        client = None  # let Chroma handle it
    
    store = Chroma.from_documents(
        documents=docs, 
        embedding=embed, 
        persist_directory=persist_dir,
        client=client,  # Pass explicit client
        collection_name=get_chroma_collection_name()
    )
    return store


def load_vectorstore():
    try:
        embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Same logic for load
        if os.environ.get("STREAMLIT_CLOUD_APP") or "streamlit.io" in os.environ.get("HOME", ""):
            client = chromadb.EphemeralClient()
            persist_dir = None
        else:
            client = None
            persist_dir = CHROMA_PERSIST_DIR
        
        store = Chroma(
            persist_directory=persist_dir, 
            collection_name=get_chroma_collection_name(), 
            embedding_function=embed,
            client=client  # Pass explicit client
        )
        return store
    except:
        return None


# Chroma clearing (local only)
def clear_chroma():
    try:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        time.sleep(0.4)
        return True
    except:
        return False
