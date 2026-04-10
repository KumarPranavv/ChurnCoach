#!/usr/bin/env python3
"""
Build (or rebuild) the FAISS vector index from the knowledge base.

Run this script whenever you update files in knowledge_base/:
    python build_faiss_index.py

The index is saved to faiss_index/ and loaded by rag_engine.py at runtime.
This script uses sentence-transformers + FAISS directly (no LangChain wrappers)
so the Streamlit app does NOT need to load sentence-transformers at runtime.
"""

import os
import glob
import pickle
import warnings

warnings.filterwarnings("ignore")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = "knowledge_base"
INDEX_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Split text into overlapping chunks with source metadata."""
    chunks = []
    # Split on section headers first
    sections = text.split("\n## ")
    all_text_parts = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section  # restore the header
        all_text_parts.append(section.strip())

    for part in all_text_parts:
        if len(part) <= chunk_size:
            chunks.append({"text": part, "source": source})
        else:
            # Slide window
            start = 0
            while start < len(part):
                end = start + chunk_size
                chunk = part[start:end]
                chunks.append({"text": chunk.strip(), "source": source})
                start = end - overlap
    return chunks


def main():
    # 1. Load knowledge base documents
    all_chunks = []
    txt_files = sorted(glob.glob(os.path.join(KB_DIR, "*.txt")))
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(fpath)
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"  📄 {source}: {len(chunks)} chunks")
    print(f"✅ Total: {len(all_chunks)} chunks from {len(txt_files)} documents")

    # 2. Generate embeddings
    print(f"⏳ Generating embeddings with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # 3. Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine since normalized)
    index.add(embeddings)
    print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dimension}")

    # 4. Save to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"✅ Saved to {INDEX_DIR}/")
    print(f"   Files: {os.listdir(INDEX_DIR)}")

    # 5. Quick verification
    query_emb = model.encode(["senior citizen high churn risk"], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb, dtype="float32"), k=2)
    print(f"\n🔍 Test search: 'senior citizen high churn risk'")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"   #{rank+1} (score={dist:.3f}, source={all_chunks[idx]['source']}):")
        print(f"       {all_chunks[idx]['text'][:100]}...")


if __name__ == "__main__":
    main()
