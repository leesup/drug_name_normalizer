from pathlib import Path

class Config:
    # Paths
    PROJ_DIR = Path(__file__).resolve().parents[2]

    RAW_DATA_DIR = PROJ_DIR / 'data' / 'raw'
    PROCESSED_DATA_DIR = PROJ_DIR / 'data' / 'processed'
    
    RAW_DB_PATH = RAW_DATA_DIR / 'chembl_35.db'
    PROCESSED_DB_PATH = PROCESSED_DATA_DIR / 'chembl_data.parquet'
    
    # Model configuration
    SAPBERT_MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
    FAISS_INDEX_PATH = PROCESSED_DATA_DIR / 'faiss_index'

    MODELS_DIR = PROJ_DIR / 'models'
    REPORTS_DIR = PROJ_DIR / 'reports'
    FIGURES_DIR = REPORTS_DIR / 'figures'
