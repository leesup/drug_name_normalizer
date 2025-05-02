# src/utils/text_utils.py

"""
Utility functions for processing text
"""

import re

def normalize_text(text):
    """
    Normalize text for consistent processing
    
    Args:
        text(str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()

    # Remove special characters except hyphens and spaces
    text = re.sub(r'[^\w\s-]', '', text)

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove specific modifiers like "tablet", "capsule" etc.
    modifiers = [
        r'\btablet(s)?\b', r'\bcapsule(s)?\b', r'\binjection(s)?\b', 
        r'\bsolution(s)?\b', r'\bsyrup(s)?\b', r'\bsuspension(s)?\b',
        r'\bcream(s)?\b', r'\bointment(s)?\b', r'\bgel(s)?\b',
        r'\b\d+\s*mg\b', r'\b\d+\s*ml\b', r'\b\d+\s*g\b'
    ]
    for modifier in modifiers:
        text = re.sub(modifier, '', text, flags=re.IGNORECASE)

    return text.strip()
