# RAG Module

A hybrid retrieval-augmented generation (RAG) pipeline with BM25 sparse search, Chroma dense embedding retrieval, and local FlashRank reranking. Designed for ingesting and querying `.txt` and `.md` knowledge bases entirely offline.

## Features

- **Hybrid retrieval** — combines BM25 (keyword) and Chroma (semantic) search for better recall
- **Local reranking** — FlashRank neural reranker runs offline with no API keys
- **Incremental ingestion** — SHA-256 change detection skips unchanged files automatically
- **Persistent indexes** — BM25 and Chroma stores survive restarts
- **Zero API calls** — everything runs locally (embeddings via sentence-transformers)
- **LangChain-native** — built on LangChain document types for easy integration

## Installation

```bash
pip install langchain-core langchain-community langchain-huggingface langchain-text-splitters rank-bm25 sentence-transformers flashrank
```

## Quick Start

### One-liner Setup

```python
from core.RAG.rag import KnowledgeRAG

rag = KnowledgeRAG(knowledge_dir="path/to/knowledge/files")
results = rag.retrieve("your question here", top_k=5)
```

### Full Pipeline

```python
from core.RAG.rag import KnowledgeRAG

# Initialize — ingests all .txt/.md files in the directory
rag = KnowledgeRAG(
    knowledge_dir="./knowledge",
    use_sparse=True,    # BM25 keyword search
    use_dense=True,     # Chroma embedding search
    use_reranker=True,  # FlashRank reranking
)

# Query
results = rag.retrieve("How does authentication work?", top_k=5)

for doc in results:
    print(f"[{doc['source']}] {doc['content'][:100]}...")
```

### Step-by-Step Pipeline

If you need more control, use the components directly:

```python
from core.RAG.chunker import load_corpus_from_directory
from core.RAG.ingestion import ingest_chunks
from core.RAG.retriever import HybridRetriever

# 1. Chunk files
corpus = load_corpus_from_directory("./knowledge")

# 2. Ingest into BM25 + Chroma
ingest_chunks(corpus)

# 3. Retrieve
retriever = HybridRetriever(use_sparse=True, use_dense=True, use_reranker=True)
results = retriever.retrieve("your query", top_k=5)
```

## API Reference

### `KnowledgeRAG` (Facade)

The main entry point. Orchestrates chunking, ingestion, and retrieval.

```python
KnowledgeRAG(
    knowledge_dir: str,           # Path to .txt/.md files
    use_sparse: bool = True,      # Enable BM25
    use_dense: bool = True,       # Enable Chroma embeddings
    use_reranker: bool = True,    # Enable FlashRank reranking
    dense_model: str = "all-MiniLM-L6-v2",  # Embedding model (reserved)
)
```

| Method | Description |
|--------|-------------|
| `retrieve(query, top_k=5)` | Search the knowledge base; returns list of dicts |

### `HybridRetriever`

The core retrieval engine. Combines sparse + dense + reranking.

```python
HybridRetriever(
    use_sparse: bool = True,
    use_dense: bool = True,
    use_reranker: bool = True,
    sparse_k: int = 10,    # BM25 fetch count
    dense_k: int = 10,     # Chroma fetch count
)
```

| Method | Description |
|--------|-------------|
| `retrieve(query, top_k=5)` | Run hybrid search with deduplication and optional reranking |

### `chunk_text(text, source, chunk_size=400, overlap=80)`

Split text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

**Returns:** `list[dict]` — each dict has `source`, `content`, and `chunk_idx`.

### `load_corpus_from_directory(directory)`

Read all `.txt` and `.md` files, chunk them, and return the corpus. Skips unchanged files via SHA-256 tracking.

**Returns:** `list[dict]` — chunked documents.

### `ingest_chunks(chunks)`

Store chunks in both Chroma (dense vectors) and BM25 (sparse index) on disk.

### `FlashRankReranker(model_name, top_k, cache_dir)`

Local neural reranker using FlashRank.

| Supported Models | Size | Notes |
|-----------------|------|-------|
| `ms-marco-TinyBERT-L-2-v2` | ~4MB | Fastest |
| `ms-marco-MiniLM-L-12-v2` | ~34MB | Balanced (default) |
| `ms-marco-MultiBERT-L-12` | ~120MB | Multilingual |
| `rank-T5-flan` | ~80MB | Best accuracy |

| Method | Description |
|--------|-------------|
| `rerank(query, documents, top_k)` | Rerank LangChain Documents; adds `rerank_score` to metadata |
| `rerank_dicts(query, chunks, top_k, content_key)` | Convenience wrapper for dict-based chunks |

### `BM25PersistentStore(path, preprocess_func)`

Persistent BM25 index backed by pickle serialization.

| Method | Description |
|--------|-------------|
| `build(documents)` | Build index from scratch and save |
| `load()` | Load existing index from disk |
| `add_documents(new_docs)` | Extend index and rebuild |

## Data Flow

```
knowledge_dir/ (.txt, .md files)
       |
       v
  [chunker.py]  SHA-256 change detection + text splitting
       |
       v
  [ingestion.py]  Store in Chroma (dense) + BM25 (sparse)
       |
       v
  data/chroma/   +   data/bm25_store.pkl
       |
       v
  [retriever.py]  BM25 + Chroma search -> dedup -> FlashRank rerank
       |
       v
  Top-k results (list of dicts)
```

## On-Disk Artifacts

| Path | Purpose |
|------|---------|
| `data/chroma/` | Chroma vector database (dense embeddings) |
| `data/bm25_store.pkl` | Pickled BM25 index |
| `data/processed_files.json` | SHA-256 hashes for incremental ingestion |

## File Overview

| File | Purpose |
|------|---------|
| `rag.py` | `KnowledgeRAG` — high-level facade orchestrating the full pipeline |
| `retriever.py` | `HybridRetriever` — combined sparse + dense retrieval with dedup |
| `chunker.py` | Text splitting and incremental file change tracking |
| `ingestion.py` | Storage into Chroma and BM25 indexes |
| `BM25.py` | `BM25PersistentStore` and `PersistentBM25Retriever` |
| `reranker.py` | `FlashRankReranker` — local neural reranking |

## Integration with Memory Module

The RAG module is also used internally by the Memory system (`core/memory/manager.py`) for intelligent memory recall. The `MemoryManager` uses `HybridRetriever` and `chunk_text` directly for in-memory retrieval without on-disk persistence.

```python
from core.RAG.retriever import HybridRetriever
from core.RAG.chunker import chunk_text

# Use RAG components for custom retrieval
retriever = HybridRetriever(use_sparse=True, use_dense=False)
# ... build an in-memory corpus and retrieve
```