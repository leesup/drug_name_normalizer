# src/models/normalizer.py

"""
Normalize drug names and assign different scores. Currently testing out various scoring methods!
"""

import numpy as np
import Levenshtein
import difflib

from src.models.retriever import Retriever
from src.utils.text_utils import normalize_text

class DrugNameNormalizer:
    def __init__(self):
        """Initialize the normalizer, which loads ChEMBL data and the retriever."""
        self.retriever = Retriever()
        self.chembl_data = self.retriever.chembl_data        
        self.all_drug_names = [normalize_text(name) for name in self.chembl_data['name_variant'].tolist()]        
        self.name_to_idx = {name: idx for idx, name in enumerate(self.all_drug_names)}
        print(f"Initialized normalizer with {len(self.all_drug_names)} drug name variants")

    def normalize(self, input_data):
        """
        Normalize drug names and return ChEMBL IDs and preferred names.
        
        Args:
            input_data: String or list of strings with drug names
            
        Returns:
            Dictionary or list of dictionaries with normalized data
        """
        if isinstance(input_data, str):
            return self._normalize_drug(input_data)
        elif isinstance(input_data, list):
            return [self._normalize_drug(name) for name in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def _normalize_drug(self, drug_name, similarity_threshold=0.85, use_hybrid=True):
        """
        Normalize a single drug name using a hybrid approach.
        
        Args:
            drug_name: The drug name to normalize
            
        Returns:
            Dictionary with normalized data
        """
        normalized_input = normalize_text(drug_name)
        
        # Exact match if the drug name is present in the ChEMBL database
        if normalized_input in self.name_to_idx:
            idx = self.name_to_idx[normalized_input]
            return {
                'drug_name': drug_name,
                'chembl_id': self.chembl_data.iloc[idx]['chembl_id'],
                'pref_name': self.chembl_data.iloc[idx]['pref_name'],
                'confidence': 1.0,
                'method': 'exact_match'
            }
        
    # Incorporate FuzzyWuzzy or other fuzzy word matching
    # Utilize FuzzyWuzzy for scoring as well