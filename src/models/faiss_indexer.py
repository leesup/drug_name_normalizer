# src/models/faiss_indexer.py

"""
Build a Faiss index from embeddings
"""

import numpy as np
import faiss

from src.config.config import Config

class FaissIndexManager:
    def __init__(self):
        self.faiss_index_path = Config.FAISS_INDEX_PATH
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

    def build_faiss_index(self, embeddings):
        """Build a Faiss index from embeddings and save."""
        print("Building FAISS index...")
        
        # Create the index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)        
        embeddings = np.array(embeddings, dtype=np.float32) # float32 is needed for Faiss
        index.add(embeddings)
        
        # Save the index
        faiss.write_index(index, str(self.faiss_index_path))
        print(f"FAISS index saved to {self.faiss_index_path}")
        
        return index

    def load_faiss_index(self):
        """Load the FAISS index."""
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}")
        
        print(f"Loading FAISS index from {self.faiss_index_path}")
        return faiss.read_index(str(self.faiss_index_path))
    