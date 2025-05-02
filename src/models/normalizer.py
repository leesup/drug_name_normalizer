# src/models/normalizer.py

from src.models.retriever import Retriever
import Levenshtein
import difflib
from src.utils.text_utils import normalize_text
import numpy as np

class DrugNameNormalizer:
    def __init__(self):
        """Initialize the normalizer, which loads ChEMBL data and the retriever."""
        self.retriever = Retriever()
        # Get the ChEMBL data from the retriever to ensure consistent data reference
        self.chembl_data = self.retriever.chembl_data
        
        # Create a list of all normalized drug names for string matching
        self.all_drug_names = [normalize_text(name) for name in self.chembl_data['name_variant'].tolist()]
        
        # For quick lookup of exact matches
        self.name_to_idx = {name: idx for idx, name in enumerate(self.all_drug_names)}
        
        print(f"Initialized normalizer with {len(self.all_drug_names)} drug name variants")

    def normalize(self, input_data, similarity_threshold=0.85, use_hybrid=True):
        """
        Normalize drug names and return ChEMBL IDs and preferred names.
        
        Args:
            input_data: String or list of strings with drug names
            similarity_threshold: Threshold for considering a match valid (0.0-1.0)
            use_hybrid: Whether to use hybrid matching (semantic + string similarity)
            
        Returns:
            Dictionary or list of dictionaries with normalized data
        """
        if isinstance(input_data, str):
            return self._normalize_drug(input_data, similarity_threshold, use_hybrid)
        elif isinstance(input_data, list):
            return [self._normalize_drug(name, similarity_threshold, use_hybrid) for name in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def _normalize_drug(self, drug_name, similarity_threshold=0.85, use_hybrid=True):
        """
        Normalize a single drug name using a hybrid approach.
        
        Args:
            drug_name: The drug name to normalize
            similarity_threshold: Threshold for considering a match valid (0.0-1.0)
            use_hybrid: Whether to use hybrid matching (semantic + string similarity)
            
        Returns:
            Dictionary with normalized data and confidence
        """
        normalized_input = normalize_text(drug_name)
        
        # Step 1: Try exact match first (fastest)
        if normalized_input in self.name_to_idx:
            idx = self.name_to_idx[normalized_input]
            return {
                'drug_name': drug_name,
                'chembl_id': self.chembl_data.iloc[idx]['chembl_id'],
                'pref_name': self.chembl_data.iloc[idx]['pref_name'],
                'confidence': 1.0,
                'method': 'exact_match'
            }
        
        # Step 2: If no exact match, try close string matches
        close_matches = self._find_close_string_matches(normalized_input)
        
        # Step 3: Use semantic search with FAISS/SapBERT
        indices, distances = self.retriever.retrieve(drug_name, k=5)
        semantic_matches = []
        
        for i, idx in enumerate(indices[0]):
            similarity = 1.0 - float(distances[0][i])  # Convert distance to similarity
            semantic_matches.append({
                'idx': int(idx),
                'similarity': similarity
            })
        
        if use_hybrid:
            # Step 4: Combine string and semantic matches for hybrid approach
            best_match = self._get_best_hybrid_match(close_matches, semantic_matches)
        else:
            # Use only semantic matches
            best_match = semantic_matches[0] if semantic_matches else None
        
        # If we have a match above the threshold
        if best_match and best_match['similarity'] >= similarity_threshold:
            idx = best_match['idx']
            return {
                'drug_name': drug_name,
                'chembl_id': self.chembl_data.iloc[idx]['chembl_id'],
                'pref_name': self.chembl_data.iloc[idx]['pref_name'],
                'confidence': best_match['similarity'],
                'method': best_match['method'] if 'method' in best_match else 'semantic'
            }
        else:
            # No good match found, return the best we have but mark low confidence
            if semantic_matches:
                idx = semantic_matches[0]['idx']
                return {
                    'drug_name': drug_name,
                    'chembl_id': self.chembl_data.iloc[idx]['chembl_id'],
                    'pref_name': self.chembl_data.iloc[idx]['pref_name'],
                    'confidence': semantic_matches[0]['similarity'],
                    'method': 'semantic_low_confidence',
                    'warning': 'Low confidence match - please verify'
                }
            else:
                return {
                    'drug_name': drug_name,
                    'chembl_id': None,
                    'pref_name': None,
                    'confidence': 0.0,
                    'method': 'no_match',
                    'warning': 'No match found'
                }

    def _find_close_string_matches(self, normalized_input, max_matches=5):
        """Find close string matches using Levenshtein distance."""
        # For very short drug names, use a smaller threshold
        threshold = 2 if len(normalized_input) <= 5 else 3
        
        # Try to find matches with small edit distance first (faster for typical typos)
        close_matches = []
        
        # First check against 100 most similar drug names (using difflib for initial filtering)
        potential_matches = difflib.get_close_matches(normalized_input, self.all_drug_names, n=100, cutoff=0.7)
        
        for name in potential_matches:
            # Calculate edit distance and normalized similarity
            distance = Levenshtein.distance(normalized_input, name)
            max_len = max(len(normalized_input), len(name))
            similarity = 1.0 - (distance / max_len)  # Normalize to 0-1 range
            
            if distance <= threshold:
                idx = self.name_to_idx[name]
                close_matches.append({
                    'idx': idx,
                    'similarity': similarity,
                    'method': 'string'
                })
                
        # Sort by similarity (highest first)
        close_matches.sort(key=lambda x: x['similarity'], reverse=True)
        return close_matches[:max_matches]

    def _get_best_hybrid_match(self, string_matches, semantic_matches):
        """
        Combine string and semantic matching approaches for better results.
        
        This weighs string matching more heavily for short drug names where
        typos are more impactful, and semantic matching more for longer names.
        """
        if not string_matches and not semantic_matches:
            return None
            
        if not string_matches:
            return semantic_matches[0]
            
        if not semantic_matches:
            return string_matches[0]
        
        # Get top match from each method
        best_string = string_matches[0]
        best_semantic = semantic_matches[0]
        
        # Check if the same drug is found by both methods
        if best_string['idx'] == best_semantic['idx']:
            # Boost confidence if both methods agree
            combined = best_string.copy()
            combined['similarity'] = max(0.95, combined['similarity'])  # At least 95% confidence when methods agree
            combined['method'] = 'hybrid_agreement'
            return combined
        
        # For strings up to 8 characters, favor string matching more
        # For longer strings, favor semantic matching more
        string_weight = 0.7 if len(self.all_drug_names[best_string['idx']]) <= 8 else 0.3
        semantic_weight = 1.0 - string_weight
            
        # Create hybrid candidates from both match sets
        hybrid_candidates = []
        
        # Process string matches
        for match in string_matches[:3]:  # Consider top 3 string matches
            hybrid_score = match['similarity'] * string_weight
            
            # Look for the same index in semantic matches to boost score
            for sem_match in semantic_matches:
                if sem_match['idx'] == match['idx']:
                    hybrid_score += sem_match['similarity'] * semantic_weight
                    break
            
            hybrid_candidates.append({
                'idx': match['idx'],
                'similarity': hybrid_score,
                'method': 'hybrid_string'
            })
            
        # Process semantic matches
        for match in semantic_matches[:3]:  # Consider top 3 semantic matches
            # Skip if already processed in string matches
            if any(c['idx'] == match['idx'] for c in hybrid_candidates):
                continue
                
            hybrid_score = match['similarity'] * semantic_weight
            
            # Look for the same index in string matches to boost score
            for str_match in string_matches:
                if str_match['idx'] == match['idx']:
                    hybrid_score += str_match['similarity'] * string_weight
                    break
            
            hybrid_candidates.append({
                'idx': match['idx'],
                'similarity': hybrid_score,
                'method': 'hybrid_semantic'
            })
        
        # Sort hybrid candidates by score
        hybrid_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        return hybrid_candidates[0] if hybrid_candidates else best_semantic
