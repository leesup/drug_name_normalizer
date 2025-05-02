# src/core/main.py

"""
Main wrapper file to run drug_name_normalizer
"""

import argparse
import pandas as pd
from pathlib import Path
from src.models.normalizer import DrugNameNormalizer

def main():
    parser = argparse.ArgumentParser(description="Drug Name Normalizer CLI")
    parser.add_argument('-d', '--drug', nargs='+', required=True,
                        help="Provide drug name(s) or path to a CSV file with a 'drug_name' column")
    parser.add_argument('-o', '--output', type=str,
                        help="Path to save the output CSV file")
    parser.add_argument('-t', '--threshold', type=float, default=0.85,
                        help="Similarity threshold for matching (0.0-1.0)")
    parser.add_argument('--string-only', action='store_true',
                        help="Use string matching only")
    parser.add_argument('--semantic-only', action='store_true',
                        help="Use semantic matching only")
    args = parser.parse_args()

    drug_inputs = args.drug
    output_path = args.output
    threshold = args.threshold
    
    # Matching methods
    use_hybrid = not (args.string_only or args.semantic_only)

    # Initialize the DrugNameNormalizer
    print("Initializing DrugNameNormalizer...")
    normalizer = DrugNameNormalizer()

    # Read the input CSV file - CSV file must include a column named "drug_name" and must be one CSV file only
    if len(drug_inputs) == 1 and drug_inputs[0].endswith(".csv"):
        csv_path = Path(drug_inputs[0])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Reading drug names from CSV: {csv_path}...")
        input_df = pd.read_csv(csv_path)

        if "drug_name" not in input_df.columns:
            raise ValueError("CSV must contain a 'drug_name' column.")

        # Apply normalization to each drug name - find a way to do this in batches if possible
        results = []
        for drug in input_df["drug_name"]:
            result = normalizer.normalize(drug, threshold, use_hybrid)
            results.append(result)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create final DataFrame with original input and results
        final_df = pd.concat([input_df.reset_index(drop=True), 
                             results_df.drop("drug_name", axis=1).reset_index(drop=True)], 
                             axis=1)

    # If individual drug name(s) are given instead of CSV
    else:
        print("Normalizing provided drug name(s)...")
        results_list = []
        for drug in drug_inputs:
            result = normalizer.normalize(drug, threshold, use_hybrid)
            results_list.append(result)

        final_df = pd.DataFrame(results_list)

    # Save results to CSV if output path is provided
    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}!")
    else:
        # Print results to terminal
        print("\nNormalization Results:")
        print(final_df.to_string(index=False))

if __name__ == "__main__":
    main()
