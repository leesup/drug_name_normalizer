# drug_name_normalizer documentation!

## Description

Normalize drug names to ChEMBL IDs and preferred names. The transformer model SapBERT (https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) and the ChEMBL database version chembl_35.sqlite.tar.gz (https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/) is used for retrieval-augmented generation (RAG).

## Commands

The Makefile contains the central entry points for common tasks related to this project.

