from __future__ import annotations
import os
import re
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 400,
    overlap: int = 80
) -> list[dict]:
    """Returns a list of dicts:
        {"source": str, "content": str, "chunk_idx": int}
        where each dict represents a chunk of the original text, along with its source and index.
        to access chunks -> chunks = chunks[i]["content"]
        """

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

#=============UTILS=============
def get_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

import json
import os

TRACKING_FILE = "data/processed_files.json"


def load_tracking():
    if not os.path.exists(TRACKING_FILE):
        return {}
    with open(TRACKING_FILE, "r") as f:
        return json.load(f)


def save_tracking(data):
    os.makedirs("data", exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)

#===============================

def load_corpus_from_directory(directory: str) -> list[dict]:
    corpus: list[dict] = []

    if not os.path.isdir(directory):
        return corpus

    tracking = load_tracking()
    updated_tracking = {}

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            file_hash = get_file_hash(text)

            if filename in tracking and tracking[filename] == file_hash:
                updated_tracking[filename] = file_hash
                continue

            chunks = chunk_text(text, source=filename)
            corpus.extend(chunks)

            updated_tracking[filename] = file_hash

        except OSError as exc:
            print(f"[Chunker] Could not read '{filepath}': {exc}")

    # Save updated tracking
    save_tracking(updated_tracking)

    return corpus