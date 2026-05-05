from RAG.reranker import FlashRankReranker  # adjust import if needed
from langchain_core.documents import Document


def generate_dummy_docs():
    return [
        Document(
            page_content="LangChain helps build LLM-powered applications.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="BM25 is a ranking algorithm used in search engines.",
            metadata={"source": "doc2"},
        ),
        Document(
            page_content="Chroma is a vector database for storing embeddings.",
            metadata={"source": "doc3"},
        ),
        Document(
            page_content="Transformers are deep learning models used in NLP.",
            metadata={"source": "doc4"},
        ),
    ]


def test_reranker():
    query = "What is a vector database?"

    print("[Test] Initializing reranker...")
    reranker = FlashRankReranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        top_k=3,
    )

    docs = generate_dummy_docs()

    print("\n[Test] BEFORE reranking:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content}")

    reranked = reranker.rerank(query, docs)

    print("\n[Test] AFTER reranking:")
    for i, doc in enumerate(reranked):
        print(f"\nRank {i+1}:")
        print("Content :", doc.page_content)
        print("Score   :", doc.metadata.get("rerank_score"))
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    test_reranker()