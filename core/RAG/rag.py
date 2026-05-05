from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.RAG.chunker import load_corpus_from_directory
from core.RAG.retriever import HybridRetriever
from core.RAG.ingestion import ingest_chunks


logger = logging.getLogger(__name__)

class KnowledgeRAG:
    """
    Loads .txt / .md files from a directory and retrieves relevant chunks
    using hybrid sparse + dense retrieval with optional LLM re-ranking.

    Parameters
    ----------
    knowledge_dir : path to the directory containing knowledge files.
    provider      : LLMProvider (enables LLM re-ranker when provided).
    use_sparse    : enable BM25 retrieval (default True).
    use_dense     : enable embedding retrieval (default True).
    use_reranker  : enable LLM re-ranking (default True; requires provider).
    dense_model   : sentence-transformers model name.
    """

    def __init__(
        self,
        knowledge_dir: str,
        use_sparse: bool = True,
        use_dense: bool = True,
        use_reranker: bool = True,
        dense_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.knowledge_dir = knowledge_dir

        self._retriever = HybridRetriever(
            use_sparse=use_sparse,
            use_dense=use_dense,
            use_reranker=use_reranker,
        )

        corpus = load_corpus_from_directory(knowledge_dir)
        ingest_chunks(corpus)
        
        logger.info(
            "[KnowledgeRAG] Loaded %d chunks from '%s'", len(corpus), knowledge_dir
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return up to *top_k* relevant chunks for *query*.

        Each result dict has at least:
            {"source": str, "content": str}
        """
        return self._retriever.retrieve(query, top_k=top_k)