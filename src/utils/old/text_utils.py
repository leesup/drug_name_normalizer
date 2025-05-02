"""
Text utility functions for drug name normalization
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
    # Convert to lowercase
    text = text.lower()

    # Remove special chracters except hyphens and spaces
    text = re.sub(r'[^\w\s-]', '', text)

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
