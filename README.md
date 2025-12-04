# AI-Powered Document Search (RAG System)

This project is a Retrieval-Augmented Generation (RAG) application designed to extract insights from PDF documents using Google Gemini, LangChain, and ChromaDB. It enables users to upload one or more documents and interact with them through question answering, summarization, and multi-document comparison.

Built with Streamlit, this system demonstrates how modern LLMs can be combined with vector databases to create intelligent document-processing tools suitable for enterprise environments.

---

## Overview

The application processes uploaded PDFs by extracting text, splitting it into meaningful chunks, generating embeddings, and storing them in a persistent vector database. Users can then query the documents, obtain summaries, or compare multiple documents in a structured format.

The focus of this project is reliability, correctness, and transparency in retrieval-based answering.

---

## Key Features

### PDF Upload & Processing
- Upload one or multiple PDFs.
- Automatic text extraction and preparation for downstream tasks.

### Intelligent Chunking
- Documents are divided into smaller chunks to improve retrieval quality and reduce hallucination.

### Vector Database (ChromaDB)
- Embeddings are stored in a persistent ChromaDB collection for fast similarity search.

### Retrieval-Augmented Question Answering
- Relevant chunks are retrieved.
- Google Gemini generates context-aware answers grounded in the documents.

### Dynamic Summarization
- Generates summaries with 5–15 bullet points depending on document size.

### Multi-Document Comparison
- Compares any number of uploaded documents, generating:
  - Individual document overviews
  - Similarities across documents
  - Differences between documents
  - A concluding summary

### Source Transparency
- Every answer includes the exact PDF(s) used in retrieval.

### Database Reset System
- Safely resets the vector database, cached embeddings, and session state.
- Ensures no stale data or outdated chunks remain.

---

## Technology Stack

- Python
- Streamlit (UI)
- LangChain (RAG orchestration)
- ChromaDB (vector store)
- Google Gemini 2.5 Flash (LLM)
- PDF parsing utilities

---

## How It Works

1. User uploads one or more PDFs.
2. Text is extracted and chunked into meaningful segments.
3. Embeddings are generated and stored in ChromaDB.
4. The retriever finds the most relevant chunks based on similarity.
5. Gemini generates the final answer, summary, or document comparison based strictly on retrieved context.

This architecture ensures accuracy, reduces hallucination, and maintains transparency.

---

## Use Cases

- Research paper analysis
- Policy document review
- Extracting insights from long reports
- Enterprise knowledge search
- Education and academic summarization
- Legal, healthcare, or financial document comparison

---

## Author

**Aditi Arya**  

---

*Feel free to contribute, raise issues, or suggest improvements!*
