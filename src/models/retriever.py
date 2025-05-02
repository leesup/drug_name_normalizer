# src/models/retriever.py

"""
Retrieve data from ChEMBL database. 
"""

import numpy as np
import pandas as pd
import time

from src.models.faiss_indexer import FaissIndexManager
from src.models.transformer_models import SapBERT
from src.data.data_loader import DataLoader

class Retriever:
    def __init__(self):
        start_time = time.time()
        print("Initializing Retriever...")
        
        self.faiss_manager = FaissIndexManager()
        index_exists = self.faiss_manager.faiss_index_path.exists() # Check if Faiss index already exists
        self.transformer = SapBERT()
        self.data_loader = DataLoader()
        self.chembl_data = self.data_loader.load_chembl_db()
        
        if index_exists:
            print(f"Loading existing FAISS index from {self.faiss_manager.faiss_index_path}")
            self.index = self.faiss_manager.load_faiss_index()
        else:
            # Generate embeddings and build the index
            if 'embeddings' not in self.chembl_data.columns:
                print("Generating embeddings for drug names...")
                embeddings = self._generate_embeddings(self.chembl_data['name_variant'].tolist())
                self.chembl_data['embeddings'] = pd.Series(list(embeddings))                
                self.chembl_data.to_parquet(self.data_loader.processed_db_path)
                print(f"Saved dataframe with embeddings to {self.data_loader.processed_db_path}")
            
            # Extract embeddings into numpy array for Faiss
            print("Preparing embeddings for FAISS index...")
            embeddings_array = np.stack(self.chembl_data['embeddings'].tolist())
            
            # Build and save the Faiss index
            print("Building new FAISS index...")
            self.index = self.faiss_manager.build_faiss_index(embeddings_array)
        
        print(f"Retriever initialization completed in {time.time() - start_time:.2f} seconds")

    def _generate_embeddings(self, texts):
        """Generate embeddings for a list of texts in batches."""
        batch_size = 64 # Will be testing out different batch sizes in the future to see if I can make it faster
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.transformer.encode(batch_texts)
            all_embeddings.extend([emb for emb in batch_embeddings])
        
        print(f"Finished generating embeddings for {len(texts)} texts.")
        return all_embeddings

    def retrieve(self, query, k=5):
        """Retrieve the top-k most similar entries."""
        if not query:
            raise ValueError("Query must be a non-empty string.")
            
        query_embedding = self.transformer.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return indices, distances
