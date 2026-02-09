# Identifying Drug Repurposing Candidates for Rare Neuro-muscular Disorders, Using Different AI Methods on the Literature Knowledge Graph

This repository contains the code and resources accompanying the work:
```
Identifying Drug Repurposing Candidates for Rare Neuro-muscular Disorders Using Different AI Methods on the Literature Knowledge Graph (2025)

Papadimas, F., Svolou, S., Bougiatiotis, K., Aisopos, F., Krithara, A., and Paliouras, G.
```
This work was conducted in the context of the **SIMPATHIC** project, funded by the European Union’s Horizon 2020 research and innovation programme (Grant Agreement **No. 101080249**).

## Overview

**Drug repurposing** is a critical yet challenging task for rare diseases, where limited patient populations and sparse curated biomedical evidence hinder traditional drug discovery pipelines. This project presents a computational framework for drug–disease link prediction, integrating heterogeneous biomedical evidence into a unified literature-based knowledge graph.

Focusing on seven rare neurological, neurometabolic, and neuromuscular disorders:

*1. SpinoCerebellar Ataxia type 3 (SCA3)*

*2. Congenital NeuroTransmitter defects (CNT)*

*3. Pyridoxine Dependent Epilepsy (PDE )*

*4. Congenital disorder glycosylation (PMM2)*

*5. Zellweger Spectrum Disorders (ZSD)*

*6. Myotonic Dystrophy type 1 (DM1)*

*7. Congenital Myasthenic Syndrome (CMS)*


we construct a **disease-centered biomedical knowledge graph** and evaluate multiple artificial intelligence approaches.

## Link Prediction Approaches

- Text-based similarity Baseline
- Rule-based inference (**AnyBURL**)
- Graph Neural Networks (**R-GCN, CompGCN**)
- Path-based methods (**Path Analysis, PAM**)

**Disease similarity** information derived from text embeddings is incorporated to enhance predictions.

## Ensemble Prediction

Our **ensemble model** aggregates predictions from the best-performing models using a weighted ranking strategy, producing consolidated drug candidate lists for each disease.

### Evaluation

- Quantitative evaluation using MRR and Hits@K

- Per-disease performance analysis

### Expert manual validation of top-ranked drug candidates

Top-ranked drug candidates for each disorder were reviewed by domain experts using a structured annotation scheme (e.g., Investigated – does not work, Treats a symptom).

