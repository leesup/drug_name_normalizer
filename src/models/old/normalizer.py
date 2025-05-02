# src/models/normalizer.py

from src.models.retriever import Retriever

class DrugNameNormalizer:
    def __init__(self):
        """Initialize the normalizer, which loads ChEMBL data and the retriever."""
        self.retriever = Retriever()
        # Get the ChEMBL data from the retriever to ensure consistent data reference
        self.chembl_data = self.retriever.chembl_data

    def normalize(self, input_data):
        """Normalize drug names and return ChEMBL IDs and preferred names."""
        if isinstance(input_data, str):
            return self._normalize_drug(input_data)
        elif isinstance(input_data, list):
            return [self._normalize_drug(name) for name in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def _normalize_drug(self, drug_name):
        """Normalize a single drug name."""
        # Use the retriever to find the best match for the drug name
        indices, _ = self.retriever.retrieve(drug_name)
        best_match_idx = indices[0][0]  # Get the first (best) match index

        # Retrieve the corresponding ChEMBL ID and preferred name
        chembl_id = self.chembl_data.iloc[best_match_idx]['chembl_id']
        pref_name = self.chembl_data.iloc[best_match_idx]['pref_name']
        
        return {'drug_name': drug_name, 'chembl_id': chembl_id, 'pref_name': pref_name}
