import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import shutil, time
from app.config import get_chroma_collection_name, CHROMA_PERSIST_DIR


def _embeddings():
    # Configure with API key explicitly – no ADC required
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )


def build_vectorstore(docs, persist: bool = True):
    embed = _embeddings()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embed,
        persist_directory=CHROMA_PERSIST_DIR if persist else None,
        collection_name=get_chroma_collection_name(),
    )
    return store


def load_vectorstore():
    try:
        embed = _embeddings()
        store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=get_chroma_collection_name(),
            embedding_function=embed,
        )
        return store
    except:
        return None


def clear_chroma():
    try:
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        time.sleep(0.4)
        return True
    except:
        return False

