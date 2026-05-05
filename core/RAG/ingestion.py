from __future__ import annotations

import os
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.RAG.BM25 import BM25PersistentStore


CHROMA_PATH = "data/chroma"
BM25_PATH   = "data/bm25_store.pkl"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def chunks_to_documents(chunks: List[dict]) -> List[Document]:
    docs = []
    for chunk in chunks:
        doc_id = str(uuid.uuid4())
        docs.append(Document(
            page_content=chunk["content"],
            metadata={
                "source":    chunk["source"],
                "chunk_idx": chunk["chunk_idx"],
                "id":        doc_id,
            },
            id=doc_id,
        ))
    return docs


def store_in_chroma(documents: List[Document]):
    if not documents:
        return

    if os.path.exists(CHROMA_PATH):
        # Load existing collection and add
        vectordb = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model,
        )
        vectordb.add_documents(documents)
    else:
        # First time — build from scratch
        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH,
        )


def store_in_bm25(documents: List[Document]):
    store = BM25PersistentStore(path=BM25_PATH)
    if os.path.exists(BM25_PATH):
        store.load()
        store.add_documents(documents)
    else:
        store.build(documents)


def ingest_chunks(chunks: List[dict]):
    if not chunks:
        print("[Ingest] No chunks to process.")
        return
    print(f"[Ingest] Processing {len(chunks)} chunks...")
    documents = chunks_to_documents(chunks)
    store_in_chroma(documents)
    store_in_bm25(documents)
    print("[Ingest] Done.")