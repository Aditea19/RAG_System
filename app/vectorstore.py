from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil, time
from app.config import get_chroma_collection_name, CHROMA_PERSIST_DIR

def build_vectorstore(docs, persist: bool = True):
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma.from_documents(
        documents=docs, 
        embedding=embed, 
        persist_directory=CHROMA_PERSIST_DIR, 
        collection_name=get_chroma_collection_name()
    )
    return store

def load_vectorstore():
    try:
        embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
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
