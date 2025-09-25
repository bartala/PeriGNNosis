# PeriGNNosis: AI Graph Model for Detecting Childbirth-Related PTSD

## Overview

Maternal mental health disorders are the most common complications of childbirth and a leading cause of maternal morbidity. 
Childbirth-related posttraumatic stress disorder (CB-PTSD) remains under-recognized in perinatal care despite lasting consequences for mothers, infants, and families.
We present PeriGNNosis (Peritraumatic Graph Neural Network Diagnosis), an artificial intelligence (AI) model that detects CB-PTSD by analyzing childbirth narratives, peritraumatic distress, and obstetric data. 
PeriGNNosis analyzed 301 postpartum women using a large language model, extracting entities and relationships from childbirth narratives to construct a heterogeneous knowledge graph (KG), enriched with peritraumatic stress symptoms and obstetric complications. 
A graph neural network trained on our KG successfully identified women at risk for CB-PTSD (F1: 0.92, Sensitivity: 0.90, Specificity: 0.95, AUC: 0.93). 
Narrative subgraph analyses revealed denser, more interconnected, and emotionally charged graphs among women with CB-PTSD, reflecting medical intervention, loss of control, and relational strain. 
PeriGNNosis demonstrates a  scalable, patient-centered approach for CB-PTSD detection.

## Key Features

KG construction & load from Neo4j into PyTorch Geometric `HeteroData`

Feature assembly for `Document` (narrative) nodes: `degree`, `embeddings`, `cb_complication`, `pdi_q1…pdi_q13` (and optional `pdi_total`)

Heterogeneous GraphSAGE classifier with train/val/test masks

Ablation: drop-one-feature retraining (macro-F1 impact)

Reproducibility: fixed seeds across frameworks

No raw PHI/PII exposure - run locally with your own data

## Quick start

Create and activate your environment (conda or venv).

`pip install -r requirements.txt`

Prepare `.env`

`ollama_graphRAG.ipynb` - Analyze narratives using LlaMa 3.1-8B and build a knowledge graph in Neo4j

Run `cbex_cbptsd.ipynb` - 
This script builds a heterogeneous knowledge graph (KG) from Neo4j, assembles document-level features (narrative embeddings, degree, obstetric complications, Peritraumatic Distress Inventory items), and trains a heterogeneous GraphSAGE classifier to detect childbirth-related PTSD (CB-PTSD). It also includes a feature ablation routine.

Outputs:

    * hetero_graph_attribute.pt – serialized PyG graph
    
    * gnn_trained_hetero_model.pth – trained model weights
    
    * ablation_test_f1_results_pdi.csv – ablation outcomes

## Privacy note:
This repo ships no data and no model weights. The code runs against your private Neo4j instance and local CSVs.

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.

## Citing
If you find this code useful for your research, please consider citing us:

```
@article{bartal2025periGNNosis,
  title={PeriGNNosis: AI Graph Model for Detecting Childbirth-Related PTSD},
  author={Bartal, Alon, and Jagodnik, Kathleen M. and Christina, T. Pham, and Dekel, Sharon},
  journal={},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={}
}
```
