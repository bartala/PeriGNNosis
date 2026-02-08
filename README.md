# PeriGNNosis

PeriGNNosis (Peritraumatic Graph Neural Network Diagnosis) is a computational framework for detecting childbirth-related posttraumatic stress disorder (CB-PTSD) by integrating unstructured childbirth narratives, peritraumatic distress responses, and obstetric context within a heterogeneous knowledge graph and graph neural network (GNN) architecture.

The framework combines large language model (LLM)–based information extraction, knowledge graph (KG) representation, and graph neural network learning to support affect-aware modeling in emotionally sensitive clinical domains.
The design emphasizes modularity, transparency, and reproducibility for research use.

---

## Repository Structure

```
PeriGNNosis/
├── KG_with_LLM.py          # LLM-based entity and relation extraction; KG construction
├── GNN.py                  # Graph neural network definition and training
├── centralities.py         # Graph centrality analyses
├── pdi_secection.py        # Peritraumatic Distress Inventory (PDI) feature processing
├── Figures.py              # Figure generation for manuscript
├── requirements.txt        # Python dependencies
├── setup.sh                # Environment setup script
├── .env.example            # Example environment configuration
└── README.md
```

---

## Methodoloy

### Narrative Processing and Entity Extraction

Participants’ childbirth narratives are processed using a large language model to extract entities and relations representing affective, clinical, and contextual information (e.g., emotions, symptoms, medical procedures, conditions). This step transforms free-text narratives into structured components suitable for graph representation while preserving affective content.

### Knowledge Graph Construction

Extracted entities and relations are stored in a heterogeneous knowledge graph using Neo4j. Each participant is represented as a `Woman` node enriched with:

* Narrative-derived graph connectivity
* Item-level Peritraumatic Distress Inventory (PDI) responses
* Obstetric complication indicators
* CB-PTSD label derived from PTSD Checklist for DSM-5 (PCL-5) cutoff scores

Non-participant nodes represent narrative concepts (e.g., Emotion, Symptom, Condition, MedicalProcedure) connected through typed semantic and affective relations inferred by the LLM.

### Graph Neural Network Modeling

The Neo4j knowledge graph is exported to a PyTorch Geometric `HeteroData` object. A heterogeneous GraphSAGE-based graph neural network is trained to classify `Woman` nodes as CB-PTSD or no CB-PTSD by jointly leveraging:

* Graph topology
* Narrative-derived structure
* Affective and clinical features

Model performance is evaluated using standard classification metrics, including F1-score, sensitivity, specificity, and area under the ROC curve (AUC).

---

## Requirements

### Software

* Python 3.9 or later
* Neo4j Desktop (local installation with a running database)
* PyTorch
* PyTorch Geometric
* LangChain
* Ollama (for local LLM inference, e.g., Llama-3.1-8B)
* Sentence Transformers

All Python dependencies are listed in `requirements.txt`.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bartala/PeriGNNosis.git
cd PeriGNNosis
```

2. Create and activate a Python environment:

```bash
python -m venv perignnosis_env
source perignnosis_env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install and start Neo4j Desktop, and ensure a local database is running.

5. Install Ollama and download a local large language model (Llama-3.1-8B).

---

## Usage

### Step 1: Knowledge Graph Construction

Run the LLM-based extraction and knowledge graph construction:

```bash
python KG_with_LLM.py
```

This script processes childbirth narratives, extracts entities and relations using an LLM, and populates a Neo4j knowledge graph enriched with clinical and affective features.

### Step 2: Graph Neural Network Training

Train the graph neural network:

```bash
python GNN.py
```

This script loads the exported graph data, trains a heterogeneous GNN, and evaluates classification performance.

### Step 3: Graph Analysis and Figure Generation (Optional)

Additional analyses and manuscript figures can be generated using:

```bash
python centralities.py
python Figures.py
```

---

## Reproducibility Notes

* The LLM-based extraction stage is separated from GNN training to avoid information leakage and ensure reproducibility.
* Knowledge graph construction is stored explicitly in Neo4j.
* Random seeds are fixed where applicable.
* The modular design allows ablation of individual data modalities (e.g., narrative-only, PDI-only).

---

## Ethical and Clinical Considerations

PeriGNNosis is a research framework intended for computational modeling and hypothesis generation. It is not a diagnostic tool and should not be used for clinical decision-making without further validation.

Narrative data is anonymized and not publically published in accordance with the institutional review board (IRB) approvals and data privacy regulations.

---

## Citation

If you use this repository, please cite the corresponding manuscript describing the PeriGNNosis framework:

```
@article{Bartal2026PeriGNNosis,
  title   = {Detection of childbirth-related posttraumatic stress disorder by integrating postpartum narratives and clinical data},
  author  = {Bartal, Alon and Jagodnik, Kathleen M. and Chan, Shira J. and Dekel, Sharon},
  journal = {Journal of Affective Disorders},
  year    = {2026},
  note    = {Under review},
}
```

