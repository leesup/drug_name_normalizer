# src/models/faiss_indexer.py

import faiss
import numpy as np
from src.config.config import Config

class FaissIndexManager:
    def __init__(self):
        # Initialize the FAISS index path
        self.faiss_index_path = Config.FAISS_INDEX_PATH
        # Make sure the parent directory exists
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

    def build_faiss_index(self, embeddings):
        """Build a FAISS index from embeddings and save it to disk."""
        print(f"Building FAISS index with {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Create the index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to the index
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)
        
        # Save the index to disk
        faiss.write_index(index, str(self.faiss_index_path))
        print(f"FAISS index saved to {self.faiss_index_path}")
        
        return index

    def load_faiss_index(self):
        """Load the FAISS index from disk."""
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}")
        
        print(f"Loading FAISS index from {self.faiss_index_path}")
        return faiss.read_index(str(self.faiss_index_path))
    