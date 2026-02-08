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
