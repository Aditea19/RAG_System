import chromadb
from chromadb.api.client import SharedSystemClient

def build_vectorstore(docs, persist: bool = True):
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear chroma system cache to fix tenant issues
    SharedSystemClient.clear_system_cache()
    
    if os.environ.get("STREAMLIT_CLOUD_APP") or "streamlit.io" in os.environ.get("HOME", ""):
        client = chromadb.EphemeralClient()
        persist_dir = None
    else:
        persist_dir = CHROMA_PERSIST_DIR if persist else None
        client = None
    
    store = Chroma.from_documents(
        documents=docs,
        embedding=embed,
        persist_directory=persist_dir,
        client=client,
        collection_name=get_chroma_collection_name()
    )
    return store

def load_vectorstore():
    try:
        embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        SharedSystemClient.clear_system_cache()

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
            client=client
        )
        return store
    except:
        return None
