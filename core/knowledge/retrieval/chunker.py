"""
Text chunking utilities.

Strategy
--------
1. Split document by paragraphs (double newline).
2. Any paragraph longer than `max_chars` is further split with a sliding
   window of size `chunk_size` and overlap `overlap`.
3. Each chunk carries metadata: source filename, chunk index, character offset.

Supported file extensions: .txt  .wmd
"""

from __future__ import annotations
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 400,
    overlap: int = 80
) -> list[dict]:
    """Returns a list of dicts:
        {"source": str, "content": str, "chunk_idx": int}"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        strip_whitespace=True,
    )

    chunks = splitter.split_text(text)

    return [
        {"source": source, "content": chunk, "chunk_idx": idx}
        for idx, chunk in enumerate(chunks)
    ]

def load_corpus_from_directory(directory: str) -> list[dict]:
    """
    Load all .txt and .md files from *directory* and return a flat list of chunks.
    """
    corpus: list[dict] = []
    if not os.path.isdir(directory):
        return corpus

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith((".txt", ".md")):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            corpus.extend(chunk_text(text, source=filename))
        except OSError as exc:
            print(f"[Chunker] Could not read '{filepath}': {exc}")

    return corpus