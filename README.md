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

## File descriptions

Run `setup.sh` -- Shell script to initialize the environment and install required dependencies.

`requirements.txt`  -- Python dependencies for running the PeriGNNosis pipeline (PyTorch, PyG, LangChain, Neo4j, etc.).

`.env` -- Template file for environment variables. Copy to .env and fill in your Neo4j credentials, database name, and local paths.

`ollama_graphRAG.ipynb` -- Analyze narratives using LlaMa 3.1-8B and build a knowledge graph in Neo4j
Notebook integrating Ollama + LangChain for graph-augmented retrieval (GraphRAG).
Uses LLMs to extract entities/relations from text.
Populates Neo4j with narrative data.
Demonstrates querying the KG with natural language prompts.
Explores how retrieval-augmented generation can assist with explainability and interactive analysis.

`cbex_cbptsd.ipynb` -- 
This script builds a heterogeneous knowledge graph (KG) from Neo4j, assembles document-level features (narrative embeddings, degree, obstetric complications, Peritraumatic Distress Inventory items), and trains a heterogeneous GraphSAGE classifier to detect childbirth-related PTSD (CB-PTSD). It also includes a feature ablation routine.

Outputs:

    * hetero_graph_attribute.pt – serialized PyG graph
    
    * gnn_trained_hetero_model.pth – trained model weights
    
    * ablation_test_f1_results_pdi.csv – ablation outcomes

## Recommended Workflow

Follow these steps to reproduce and extend **PeriGNNosis**:

1. **Setup Environment**

   * Clone this repo
   * Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```
   * Copy `.env.example` → `.env` and fill in your Neo4j credentials (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).

2. **Prepare Data**

   * Add structured variables (`cb_complication`, `pdi_q1–q13`, etc.) to `Document` nodes in Neo4j.
   * Run the provided Cypher script in `cbex_cbptsd.ipynb` to attach PDI items and obstetric complications to narratives.

3. **Construct Knowledge Graph (KG)**

   * Use `ollama_graphRAG.ipynb` to:

     * Extract entities & relations with LLMGraphTransformer
     * Build heterogeneous KG in Neo4j
     * Export KG as a PyG `HeteroData` object (`hetero_graph_attribute.pt`)

4. **Train Graph Neural Network (GNN)**

   * Run the training loop in `cbex_cbptsd.ipynb`
   * Default model = 2-layer **GraphSAGE** (hidden=64)
   * Training outputs include accuracy, F1, and confusion matrix

5. **Evaluate Performance**

   * Expected results (replicating the paper):

     * **F1-score**: ~0.92
     * **Sensitivity**: ~0.90
     * **Specificity**: ~0.95
     * **AUC**: ~0.93

6. **Ablation Studies**

   * Use `cbex_cbptsd.ipynb` to selectively drop features (e.g., `pdi_q1`, `textEmbedding`)
   * Compare F1-score impact to identify most predictive PDI items

7. **Extend**

   * Swap in other GNN layers (e.g., GAT, RGCN, HGT)
   * Add additional structured data (EMR, demographics, physiology)
   * Fine-tune the LLM-based entity extraction on childbirth-specific corpora


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
