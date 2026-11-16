import os
import pickle
import numpy as np
import faiss

from utils.embeddings import EMBEDDING_DIM

VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
TEXTS_PATH = os.path.join(VECTOR_DIR, "stored_texts.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)


def _load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(EMBEDDING_DIM)


def _load_texts():
    if os.path.exists(TEXTS_PATH):
        with open(TEXTS_PATH, "rb") as f:
            return pickle.load(f)
    return []


def _save_index(index):
    faiss.write_index(index, INDEX_PATH)


def _save_texts(texts):
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)


def add_to_faiss(text, embedding, metadata=None):
    """Store a generated piece of content + its embedding for future retrieval."""
    index = _load_index()
    texts = _load_texts()

    vector = np.array([embedding]).astype("float32")
    index.add(vector)

    entry = {"text": text}
    if metadata:
        entry.update(metadata)
    texts.append(entry)

    _save_index(index)
    _save_texts(texts)


def search_similar(embedding, k=3):
    """Return the text of the k most similar previously-generated entries."""
    index = _load_index()
    texts = _load_texts()

    if index.ntotal == 0:
        return []

    vector = np.array([embedding]).astype("float32")
    k = min(k, index.ntotal)
    _distances, indices = index.search(vector, k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(texts[idx]["text"])
    return results


def get_all_history():
    """Return all stored entries, most recent first."""
    return list(reversed(_load_texts()))