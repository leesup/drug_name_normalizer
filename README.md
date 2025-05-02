# drug_name_normalizer

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Normalize drug names to ChEMBL IDs and preferred names. The transformer model SapBERT (https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and the ChEMBL database version chembl_35.sqlite.tar.gz (https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/) are used for retrieval-augmented generation (RAG).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml    <- The environment file for reproducing the analysis environment
│
├── __main__.py
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py           
    │
    ├── config
    │   ├── __init__.py         
    │   └── config.py           
    │
    ├── core
    │   ├── __init__.py         
    │   └── main.py
    │
    ├── data
    │   ├── __init__.py         
    │   └── data_loader.p
    │
    ├── models
    │   ├── __init__.py         
    │   ├── faiss_indexer.py
    │   ├── normalizer.py
    │   ├── retriever.py
    │   └── transformer_models.py
    │
    └── utils
        ├── __init__.py
        └── text_utils.py
             

