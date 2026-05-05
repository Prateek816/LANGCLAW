from __future__ import annotations

import logging
from pprint import pprint

from RAG.rag import KnowledgeRAG


logging.basicConfig(level=logging.INFO)


def test_knowledge_rag():
    """
    Basic integration test for KnowledgeRAG
    """

    # 👇 your test_data directory
    knowledge_dir = "test_data"

    rag = KnowledgeRAG(
        knowledge_dir=knowledge_dir,
        use_sparse=True,
        use_dense=True,
        use_reranker=True,   # set False if reranker not configured
    )

    # test queries
    queries = [
        "BM25 is a keyword-based retrieval algorithm."
    ]

    for q in queries:
        print("\n" + "=" * 50)
        print(f"Query: {q}")
        print("=" * 50)

        results = rag.retrieve(q, top_k=3)

        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source : {res.get('source')}")
            print(f"Content: {res.get('content')[:200]}...")  # preview


if __name__ == "__main__":
    test_knowledge_rag()