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
        """Build a FAISS index from embeddings."""
        # Check if the index already exists
        if self.faiss_index_path.exists():
            print(f"FAISS index already exists at {self.faiss_index_path}. Loading it...")
            return self._load_faiss_index()

        print("Building new FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)
    
        # Save the newly built index - convert Path to string
        faiss.write_index(index, str(self.faiss_index_path))
        print(f"FAISS index saved to {self.faiss_index_path}.")
        
        return index

    def _load_faiss_index(self):
        """Load the FAISS index."""
        return faiss.read_index(str(self.faiss_index_path))
    