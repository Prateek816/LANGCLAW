from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """
    Local reranker using FlashRank — no API key, runs fully offline.

    Supported models:
        "ms-marco-TinyBERT-L-2-v2"   → fastest, least accurate  (~4MB)
        "ms-marco-MiniLM-L-12-v2"    → balanced  (default)      (~34MB)
        "ms-marco-MultiBERT-L-12"    → multilingual              (~120MB)
        "rank-T5-flan"               → best accuracy, slowest    (~80MB)

    Models are downloaded automatically on first use and cached locally.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        top_k: int = 5,
        cache_dir: Optional[str] = None,
    ):
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError("Run: pip install flashrank")

        self.top_k = top_k
        self._model_name = model_name

        logger.info(f"[Reranker] Loading FlashRank model: {model_name}")
        self._ranker = Ranker(
            model_name=model_name,
            cache_dir=cache_dir or ".cache/flashrank",
        )
        logger.info("[Reranker] Model ready.")

    # ── Core ──────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Rerank a list of LangChain Documents against a query.

        Args:
            query:     The user's query string.
            documents: Candidate Documents from sparse + dense retrieval.
            top_k:     How many to return. Falls back to self.top_k.

        Returns:
            Reranked list of Documents, best match first.
            Each Document gets a `rerank_score` field added to its metadata.
        """
        if not documents:
            return []

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(documents))

        try:
            from flashrank import RerankRequest

            # FlashRank expects: [{"id": int, "text": str, "meta": dict}, ...]
            passages = [
                {
                    "id":   idx,
                    "text": doc.page_content,
                    "meta": doc.metadata,
                }
                for idx, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)

            # results is a list of dicts sorted by score descending:
            # [{"id": int, "text": str, "score": float, "meta": dict}, ...]
            reranked: List[Document] = []
            for result in results[:k]:
                original_doc = documents[result["id"]]
                # Attach rerank score to metadata for transparency
                enriched_metadata = {
                    **original_doc.metadata,
                    "rerank_score": round(float(result["score"]), 4),
                }
                reranked.append(
                    Document(
                        page_content=original_doc.page_content,
                        metadata=enriched_metadata,
                    )
                )

            return reranked

        except Exception as exc:
            logger.warning(
                f"[Reranker] FlashRank failed ({exc}). Falling back to original order."
            )
            return documents[:k]

    def rerank_dicts(
        self,
        query: str,
        chunks: List[dict],
        top_k: Optional[int] = None,
        content_key: str = "content",
    ) -> List[dict]:
        """
        Convenience method: rerank raw dicts (from retriever.retrieve())
        instead of LangChain Documents.

        Args:
            query:       The user's query string.
            chunks:      List of dicts with at least a `content` key.
            top_k:       How many to return.
            content_key: Key in the dict that holds the text (default: "content").

        Returns:
            Reranked list of dicts, best match first.
            Each dict gets a `rerank_score` field added.
        """
        if not chunks:
            return []

        # Convert dicts → Documents
        docs = [
            Document(
                page_content=chunk[content_key],
                metadata={k: v for k, v in chunk.items() if k != content_key},
            )
            for chunk in chunks
        ]

        reranked_docs = self.rerank(query, docs, top_k=top_k)

        # Convert back to dicts
        return [
            {"content": doc.page_content, **doc.metadata}
            for doc in reranked_docs
        ]