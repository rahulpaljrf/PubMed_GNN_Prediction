# PubMed Patient Graph Learning with Hierarchical Attention and GNNs

This repository contains the code for building a patient similarity graph
from PubMed abstracts using BioBERT, hierarchical multi-head attention,
and Graph Neural Networks for multi-label classification.

## Pipeline Overview
1. Data preparation and label processing
2. Chunk-based BioBERT embedding extraction
3. Hierarchical multi-head attention aggregation
4. Patient graph construction using cosine similarity
5. GNN-based multi-label node classification

## Repository Structure
- `scripts/` – main training and preprocessing scripts
- `configs/` – hyperparameters and paths
- `results/` – evaluation outputs

## Data
Due to size constraints, datasets and embeddings are not included.
Please download the PubMed dataset separately and follow `scripts/01_prepare_data.py`.

## Requirements
See `requirements.txt`.

## Citation
If you use this code, please cite our work.

