# PeriGNNosis: A Peritraumatic Graph Neural Network Diagnosis Model for Detecting Childbirth PTSD Using Graphs Based on Large Language Models

## Overview

Background: Maternal mental health disorders are the leading complications of childbirth and known cause of maternal death. Undetected disorders can also result in substantial medical maternal and infant costs. Approximately one third of postpartum experience childbirth as highly stressful and are at risk for subsequently developing childbirth-related posttraumatic stress disorder (CB-PTSD), warranting the development of accurate screening tools.
Objective: This study examined the utility of artificial intelligence (AI) based models of childbirth narrative data combined with questionnaire data of patients’ acute stress during childbirth and presence of obstetric complication for detecting CB-PTSD. We introduce PeriGNNosis, a novel AI-based model to analyze narratives using a large language model (LLM), and we constructed a heterogeneous knowledge graph (KG) representing narratives and questionnaire responses. A graph neural network (GNN) model was trained on our KG to forecast CB-PTSD.
Study Design: A total of 1,127 women provided a short, unstructured narrative of their recent childbirth  with a focus on the distressing aspects of the experience. They were also assessed for acute stress responses (Peritraumatic Distress Inventory,PDI) and for CB-PTSD symptoms (PTSD Checklist for DSM-5, PCL-5). Narrative features were processed using an LLM to build a heterogeneous knowledge graph (KG), which was then used to train the PeriGNNosis model. The graph included narrative nodes, a binary indicator of birth complications, and selected PDI items.
Results: PeriGNNosis achieved high classification performance, with a macro F1-score of 0.92, sensitivity of 0.90, and specificity of 0.95. Classification was based on graph degree, birth complications, and seven PDI items: Q2 (helplessness), Q3 (sadness/grief), Q5 (guilt), Q6 (shame), Q7 (physical reactivity), Q11 (loss of control), and Q12 (fear of death).
Conclusions: Our findings demonstrate that AI-based computational methods that use personal childbirth narratives combined with patients’ recalled stress responses can accurately identify women endorsing high levels of CB-PTSD symptoms. This model may offer a scalable, low-cost, patient-friendly tool that could be integrated into perinatal care for early screening of women following traumatic childbirth to reduce a preventable form of maternal morbidity. Further research is needed to advance and validate AI-driven models for predicting early signs of maternal psychiatric conditions.

## Quick start

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.

## Citing
If you find this code useful for your research, please consider citing us:

```
@article{jagodnik2024persistence,
  title={PeriGNNosis: A Peritraumatic Graph Neural Network Diagnosis Model for Detecting Childbirth PTSD Using Graphs Based on Large Language Models},
  author={Bartal, Alon, and Jagodnik, Kathleen M. and Dekel, Sharon},
  journal={},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={}
}
```
