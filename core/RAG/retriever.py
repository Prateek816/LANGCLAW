from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.RAG.BM25 import BM25PersistentStore, PersistentBM25Retriever
from core.RAG.reranker import FlashRankReranker


#later get paths from config.py
CHROMA_PATH = "data/chroma"
BM25_PATH   = "data/bm25_store.pkl"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class HybridRetriever:
    """
    Combines BM25 (sparse) + Chroma (dense) retrieval.
    Deduplicates results, then returns top_k.
    Plug in a reranker later via the `reranker` param.
    """

    def __init__(
        self,
        use_sparse: bool = True,
        use_dense: bool = True,
        use_reranker:bool = True,              # plug in your reranker here later
        sparse_k: int = 10,
        dense_k: int = 10,
    ):

        # ── Sparse (BM25) ─────────────────────────────────────────────
        self._sparse_retriever: Optional[PersistentBM25Retriever] = None
        if use_sparse and os.path.exists(BM25_PATH):
            store = BM25PersistentStore(path=BM25_PATH)
            store.load()
            self._sparse_retriever = PersistentBM25Retriever(store=store, k=sparse_k)
        elif use_sparse:
            print("[HybridRetriever] BM25 index not found. Run ingest_chunks() first.")

        # ── Dense (Chroma) ────────────────────────────────────────────
        self._dense_retriever: Optional[Chroma] = None
        if use_dense and os.path.exists(CHROMA_PATH):
            self._dense_retriever = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_model,
            )
        elif use_dense:
            print("[HybridRetriever] Chroma store not found. Run ingest_chunks() first.")

        # ── Reranker (FlashRank) ───────────────────────────────────────
        self._reranker: Optional[FlashRankReranker] = None
        if use_reranker:
            try:
                self._reranker = FlashRankReranker()
            except:
                print("[HybridRetriever] FlashRank not found.")
        
        

    # ── Core retrieve ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k relevant chunks.
        Returns list of dicts: {"content": str, "source": str, "chunk_idx": int, ...}
        """
        fetch_k = max(top_k * 3, top_k + 5)
        candidates: List[Document] = []

        # Sparse
        if self._sparse_retriever:
            sparse_results = self._sparse_retriever._get_relevant_documents(
                query, top_k=fetch_k
            )
            candidates.extend(sparse_results)

        # Dense
        if self._dense_retriever:
            dense_results = self._dense_retriever.similarity_search(query, k=fetch_k)
            candidates.extend(dense_results)

        if not candidates:
            return []

        # Deduplicate
        seen = set()
        unique_candidates: List[Document] = []
        for doc in candidates:
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(doc)

        # Rerank or truncate
        if self._reranker:
            reranked_docs = self._reranker.rerank(query, unique_candidates, top_k)
        else:
            reranked_docs = unique_candidates[:top_k]

        # Document → dict
        results = []
        for doc in reranked_docs:
            data = {"content": doc.page_content, **doc.metadata}
            data.pop("_idx", None)
            results.append(data)

        return results