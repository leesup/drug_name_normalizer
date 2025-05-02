# src/models/transformer_models.py

"""
Select a transformer model used for the baseline ML model.

I only tested out SapBERT for now - may need to test out other transformers.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config.config import Config

class SapBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.SAPBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(Config.SAPBERT_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to GPU if available
        print(f"SapBERT model loaded on {self.device}")

    def encode(self, texts):
        """Encode a list of texts into embeddings."""
        if not texts:
            return np.array([])
        
        # Tokenize inputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings
