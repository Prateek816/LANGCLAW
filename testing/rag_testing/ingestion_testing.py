from RAG.ingestion import ingest_chunks, CHROMA_PATH, BM25_PATH
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from RAG.BM25 import BM25PersistentStore


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def generate_dummy_chunks():
    return [
        {
            "content": "LangChain is a framework for building LLM applications.",
            "source": "doc1.txt",
            "chunk_idx": 0,
        },
        {
            "content": "Chroma is a vector database used for embeddings.",
            "source": "doc2.txt",
            "chunk_idx": 1,
        },
        {
            "content": "BM25 is a ranking function used in information retrieval.",
            "source": "doc3.txt",
            "chunk_idx": 2,
        },
    ]


# -----------------------------
# 🔍 Inspect Chroma
# -----------------------------
def inspect_chroma():
    print("\n[Inspect] Checking Chroma DB...")

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )

    collection = vectordb._collection.get(include=["documents", "metadatas"])

    docs = collection["documents"]
    metas = collection["metadatas"]

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"\nDoc {i+1}:")
        print("Content :", doc)
        print("Metadata:", meta)


# -----------------------------
# 🔍 Inspect BM25
# -----------------------------
def inspect_bm25():
    print("\n[Inspect] Checking BM25 Store...")

    store = BM25PersistentStore(path=BM25_PATH)
    store.load()

    for i, doc in enumerate(store.docs):
        print(f"\nDoc {i+1}:")
        print("Content :", doc.page_content)
        print("Metadata:", doc.metadata)


# -----------------------------
# 🚀 Test
# -----------------------------
def test_ingestion():
    chunks = generate_dummy_chunks()
    print("[Test] Starting ingestion test...")

    ingest_chunks(chunks)

    print("\n[Test] Ingestion completed successfully!")

    # 👇 inspect both stores
    inspect_chroma()
    inspect_bm25()


if __name__ == "__main__":
    test_ingestion()