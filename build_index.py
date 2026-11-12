"""Build the Chroma index from the documents.

Usage:  python build_index.py [chunk_size] [overlap]
The embeddings run on your own computer, so no API key is needed here.
The first run downloads the small Hugging Face model once.
"""
import sys

from rag_pipeline import build_index, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNK_OVERLAP

_, count = build_index(size, overlap)
print(f"Index rebuilt: {count} chunks (chunk_size={size}, overlap={overlap})")
