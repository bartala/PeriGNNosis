# PeriGNNosis: AI Graph Model for Detecting Childbirth-Related PTSD

## Overview

Childbirth-related posttraumatic stress disorder (CB-PTSD) can follow stressful or medically complicated births but often goes unrecognized.
Narrative accounts of childbirth provide rich affective information that may complement clinical screening tools.
We developed an affect-aware machine learning framework to identify probable CB-PTSD by integrating postpartum narratives with peritraumatic distress and obstetric context.
In a web-based survey, 301 postpartum women (mean 2.5 months postpartum) provided a written childbirth narrative ($\ge$30 words), responses to the Peritraumatic Distress Inventory (PDI), and obstetric information.
Probable CB-PTSD was defined as a PTSD Checklist for DSM-5 (PCL-5) score $\ge$32.
A large language model extracted entities and relations from narratives to construct a heterogeneous knowledge graph linking narrative content with PDI symptoms and obstetric complications.
A heterogeneous graph neural network classifier was evaluated using stratified nested cross-validation.
Across held-out test folds, the model achieved a mean area under the curve (AUC) of 0.87 and a mean F1-score of 0.76, outperforming clinical (PDI-based) and text-only baselines.
Feature-stability analyses identified a parsimonious subset of peritraumatic distress symptoms capturing emotional threat and somatic arousal.
Knowledge graph analyses revealed that narratives of women with CB-PTSD exhibited higher local connectivity and altered centrality distributions, indicating differences in narrative organization rather than overrepresentation of specific semantic categories.
Integrating childbirth narratives with peritraumatic distress and obstetric context provides complementary information for CB-PTSD risk stratification.
Narrative-derived knowledge graphs offer a scalable and interpretable framework for postpartum mental health screening, warranting further evaluation in prospective and clinical settings.

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
