# src/core/main.py

import argparse
import pandas as pd
from pathlib import Path
from src.models.normalizer import DrugNameNormalizer
from src.utils.text_utils import normalize_text

def main():
    parser = argparse.ArgumentParser(description="Drug Name Normalizer CLI")
    parser.add_argument('-d', '--drug', nargs='+', required=True,
                        help="Provide drug name(s) or path to a CSV file with a 'drug_name' column")
    parser.add_argument('-o', '--output', type=str,
                        help="Path to save the output CSV file")
    args = parser.parse_args()

    drug_inputs = args.drug
    output_path = args.output

    # Initialize the normalizer
    print("Initializing DrugNameNormalizer...")
    normalizer = DrugNameNormalizer()

    # Case 1: if input is a path to a CSV file
    if len(drug_inputs) == 1 and drug_inputs[0].endswith(".csv"):
        csv_path = Path(drug_inputs[0])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Reading drug names from CSV: {csv_path}...")
        input_df = pd.read_csv(csv_path)

        if "drug_name" not in input_df.columns:
            raise ValueError("CSV must contain a 'drug_name' column.")

        # Apply text normalization
        input_df["cleaned_name"] = input_df["drug_name"].apply(normalize_text)
        
        # Apply normalization to each drug name
        results = []
        for drug in input_df["cleaned_name"]:
            result = normalizer.normalize(drug)
            results.append(result)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create final DataFrame with original input and results
        final_df = pd.concat([input_df, results_df[["chembl_id", "pref_name"]]], axis=1)

    # Case 2: input is one or more drug names
    else:
        print("Normalizing provided drug name(s)...")
        results_list = []
        for drug in drug_inputs:
            cleaned = normalize_text(drug)
            result = normalizer.normalize(cleaned)
            results_list.append({
                "drug_name": drug,
                "chembl_id": result['chembl_id'],
                'pref_name': result['pref_name']
            })

        final_df = pd.DataFrame(results_list)

    # Save results to CSV if output path is provided
    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}!")
    else:
        # Print results to console
        print("\nNormalization Results:")
        print(final_df.to_string(index=False))

if __name__ == "__main__":
    main()
