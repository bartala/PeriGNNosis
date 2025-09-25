# PeriGNNosis: AI Graph Model for Detecting Childbirth-Related PTSD

## Overview

Maternal mental health disorders are the most common complications of childbirth and a leading cause of maternal morbidity. 
Childbirth-related posttraumatic stress disorder (CB-PTSD) remains under-recognized in perinatal care despite lasting consequences for mothers, infants, and families.
We present PeriGNNosis (Peritraumatic Graph Neural Network Diagnosis), an artificial intelligence (AI) model that detects CB-PTSD by analyzing childbirth narratives, peritraumatic distress, and obstetric data. 
PeriGNNosis analyzed 301 postpartum women using a large language model, extracting entities and relationships from childbirth narratives to construct a heterogeneous knowledge graph (KG), enriched with peritraumatic stress symptoms and obstetric complications. 
A graph neural network trained on our KG successfully identified women at risk for CB-PTSD (F1: 0.92, Sensitivity: 0.90, Specificity: 0.95, AUC: 0.93). 
Narrative subgraph analyses revealed denser, more interconnected, and emotionally charged graphs among women with CB-PTSD, reflecting medical intervention, loss of control, and relational strain. 
PeriGNNosis demonstrates a  scalable, patient-centered approach for CB-PTSD detection.

## Quick start

`ollama_graphRAG.ipynb` - Analyze narratives using LlaMa and build a knowledge graph in Neo4j

`cbex_cbptsd.ipynb` - Analyze the knowledge graph and train PeriGNNosis


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
