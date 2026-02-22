# PeriGNNosis

PeriGNNosis (Peritraumatic Graph Neural Network Diagnosis) is a research framework for detecting probable childbirth-related posttraumatic stress disorder (CB-PTSD) by integrating:

- postpartum childbirth narratives (free text)
- item-level peritraumatic distress (PDI)
- obstetric complication context

The pipeline constructs a heterogeneous knowledge graph (KG) from narratives using an LLM-based extraction step, embeds nodes with a long-context sentence embedding model, and trains a leakage-aware heterogeneous GNN to classify participant (`Woman`) nodes.

This repository accompanies the manuscript:

- **Detection of childbirth-related posttraumatic stress disorder by integrating postpartum narratives and clinical data**
- **Journal of Affective Disorders** (under review)

## Method summary

1. **LLM-based extraction**: An LLM converts each narrative into entities and relations (affective, clinical, contextual).
2. **KG storage**: Entities/relations are stored in **Neo4j** as a heterogeneous graph; each participant is a `Woman` node.
3. **Node embeddings**:
   - `Woman` nodes: embedding of the full narrative text
   - non-`Woman` nodes: embedding of the entity surface text
4. **Feature fusion for classification**:
   - `Woman.x = [narrative_embedding, PDI_items, obstetric_complication_flag]`
5. **Leakage-aware evaluation**:
   - nested cross-validation with fold splits at the `Woman` level
   - held-out `Woman` nodes are prevented from contributing outgoing messages during GNN training
   - threshold selection is performed after model/feature selection using out-of-fold predictions

## Repository contents

- `KG_with_LLM.py`  
  Builds the Neo4j KG from narratives using a local LLM, computes embeddings for all nodes, and exports the KG to PyTorch Geometric (`HeteroData`).

- `GNN.py`  
  Trains/evaluates PeriGNNosis and baselines using nested cross-validation with leakage controls, and performs threshold selection on pooled out-of-fold predictions.

- `centralities.py`  
  Utilities for graph centrality analyses (used for manuscript graph-structure comparisons).

- `Figures.py`  
  Figure generation utilities used in the manuscript.

- `requirements.txt`, `setup.sh`, `.env.example`

## Requirements

### Software
- Python 3.9+
- Neo4j Desktop (local Neo4j instance running)
- Ollama (local LLM inference; e.g., Llama 3.1 8B)
- PyTorch + PyTorch Geometric
- LangChain
- sentence-transformers

### Hardware (typical)
- CPU-only is possible but slow for extraction/embedding
- A CUDA-capable GPU is recommended for faster embedding and GNN training

## Installation

```bash
git clone https://github.com/bartala/PeriGNNosis.git
cd PeriGNNosis

python -m venv perignnosis_env
source perignnosis_env/bin/activate

pip install -r requirements.txt
````

## Data

Create a local `data/` folder with at least these files:

* `data/embedded_CBPTSD.csv`
* `data/CBEx_pdi.csv`

### Expected columns

`embedded_CBPTSD.csv` must include:

* `record_id` (unique participant id)
* `source` (the code filters to `CBEx`)
* `narrative` (free-text childbirth narrative)
* `CB_PTSD` (0/1 label)
* `obstetric_complication` (0/1)

`CBEx_pdi.csv` must include:

* `record_id`
* PDI item columns named `pdi_*` (e.g., `pdi_1 ... pdi_13`)

The KG construction script excludes narratives shorter than `MIN_WORDS` (default: 30).

## Quickstart

### Start Neo4j

Start your local Neo4j database (Neo4j Desktop). Confirm credentials in `KG_with_LLM.py` (or adapt to use `.env`).

### Start Ollama + pull the LLM

```bash
ollama pull llama3.1:8b
```

### Build the KG + compute node embeddings + export to PyG

```bash
python KG_with_LLM.py
```

Outputs:

* `periGNNosis_graph.pt` (PyTorch Geometric `HeteroData`)
* `periGNNosis_metadata.csv` (filtered cohort table used for modeling)

### Train/evaluate models (nested CV, leakage-aware)

```bash
python GNN.py
```

`GNN.py` implements a nested CV pipeline (outer folds for evaluation; inner folds for model selection/PDI subset selection as configured in the script) and selects an operating threshold using pooled out-of-fold predictions after model selection.

## Reproducibility notes

* The LLM extraction stage is separated from GNN training.
* Random seeds are set where applicable; however, LLM-based extraction may still vary depending on the local inference stack.
* Leakage controls are applied at the `Woman` level during training/evaluation.

## Ethical and clinical disclaimer

PeriGNNosis is a research framework intended for computational modeling and hypothesis generation. It is not a diagnostic device and must not be used for clinical decision-making without prospective validation, clinical governance, and appropriate regulatory review.

## Citation

If you use this repository, please cite the corresponding manuscript:

```bibtex
@article{Bartal2026PeriGNNosis,
  title   = {Detection of childbirth-related posttraumatic stress disorder by integrating postpartum narratives and clinical data},
  author  = {Bartal, Alon and Jagodnik, Kathleen M. and Chan, Shira J. and Dekel, Sharon},
  journal = {Journal of Affective Disorders},
  year    = {2026},
  note    = {Under review}
}
```
