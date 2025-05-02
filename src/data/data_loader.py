# src/data/data_loader.py

import sqlite3
import pandas as pd
from pathlib import Path
from src.config.config import Config

class DataLoader():
    def __init__(self):
        self.raw_db_path = Config.RAW_DB_PATH
        self.processed_db_path = Config.PROCESSED_DB_PATH
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
    
    def load_chembl_db(self):
        if self.processed_db_path.exists():
            print("Processed data already exists. Loading from parquet file...")
            return pd.read_parquet(self.processed_db_path)

        # Ensure the processed data directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        print("Processing ChEMBL data...")
        try:
            with sqlite3.connect(self.raw_db_path) as conn:
                query = """
                SELECT DISTINCT 
                    md.chembl_id, md.pref_name, ms.synonyms
                FROM 
                    molecule_dictionary md
                LEFT JOIN
                    molecule_synonyms ms ON md.molregno = ms.molregno
                WHERE
                    md.molecule_type = 'Small molecule'
                    AND md.pref_name IS NOT NULL
                """
                # Extract data
                df = pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error while connecting to database: {e}")
            raise

        # Process the data
        processed_data = self._process_data(df)
        
        # Convert to DataFrame and return
        processed_df = pd.DataFrame(processed_data)

        # Save the processed DataFrame to a parquet file.
        processed_df.to_parquet(self.processed_db_path)
        print(f"Processed data saved to {self.processed_db_path}!")

        return processed_df

    def _process_data(self, df):
        """Process the raw data into the desired format."""
        processed_data = []

        for _, row in df.iterrows():
            chembl_id = row['chembl_id']
            pref_name = row['pref_name']
            
            # Add the preferred name as a mapping
            processed_data.append({
                'name_variant': pref_name,
                'chembl_id': chembl_id,
                'pref_name': pref_name
            })

            # Add synonyms as mappings
            if pd.notna(row['synonyms']):
                synonyms = row['synonyms'].split('|')
                for synonym in synonyms:
                    synonym = synonym.strip()  # Clean up extra spaces
                    if synonym:  # Ensure it's not an empty string
                        processed_data.append({
                            'name_variant': synonym,
                            'chembl_id': chembl_id,
                            'pref_name': pref_name
                        })

        return processed_data
