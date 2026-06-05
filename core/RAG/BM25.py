from __future__ import annotations

import logging
import os
import pickle
from typing import Callable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
from rank_bm25 import BM25Okapi


def default_preprocessing_func(text: str) -> List[str]:
    return text.lower().split()


class BM25PersistentStore:
    def __init__(
        self,
        path: str,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
    ):
        self.path = path
        self.preprocess_func = preprocess_func
        self.docs: List[Document] = []
        self.vectorizer: Optional[BM25Okapi] = None

    def build(self, documents: List[Document]):
        logger.info("Building BM25 index with %d documents", len(documents))
        self.docs = documents
        tokenized = [self.preprocess_func(doc.page_content) for doc in documents]
        self.vectorizer = BM25Okapi(tokenized)
        self._save()

    def load(self):
        if not os.path.exists(self.path):
            logger.error("No BM25 index found at %s", self.path)
            raise FileNotFoundError(f"No BM25 index found at {self.path}")
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.vectorizer = data["vectorizer"]
        logger.debug("Loaded BM25 index with %d documents from %s", len(self.docs), self.path)

    def add_documents(self, new_docs: List[Document]):
        if not self.docs:
            self.build(new_docs)
            return
        logger.debug("Adding %d documents to BM25 index", len(new_docs))
        self.docs.extend(new_docs)
        tokenized = [self.preprocess_func(doc.page_content) for doc in self.docs]
        self.vectorizer = BM25Okapi(tokenized)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"docs": self.docs, "vectorizer": self.vectorizer}, f)
        logger.debug("Saved BM25 index (%d documents) to %s", len(self.docs), self.path)


class PersistentBM25Retriever:
    # Plain class — no Pydantic, no Field()
    def __init__(self, store: BM25PersistentStore, k: int = 4):
        self.store = store
        self.k = k

    def _get_relevant_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if self.store.vectorizer is None:
            raise ValueError("BM25 index not initialized. Call build() or load() first.")
        tokens = self.store.preprocess_func(query)
        k = top_k if top_k is not None else self.k
        return self.store.vectorizer.get_top_n(tokens, self.store.docs, n=k)