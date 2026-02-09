## PAM-based Methodology

This repository contains the accompanying code for running the   [PAM-based methodology](https://github.com/kbogas/PAM_BoP) for **link prediction** in the **drug–disease** setting.

---

## Usage

### Installation

First, install the required dependencies:

```cmd
pip install -r requirements.txt
```

### Path-Based Feature Generation

To generate path-based features, run:

```cmd
python pam_feature_ectraction.py -k 2 -feat 500
```

This command extracts features from 2-hop Path Aggregation Modules (PAMs) and retains up to 500 features per node.


### Evaluation

To evaluate the model, run:


```cmd
python pam_eval.py
```

This script produces:
1. The **top-10 predicted drugs** per disease
2. Aggregate **evaluation metrics** averaged across all unique diseases (single run)

Example output:

```cmd
           num_train_feats          mr       mrr         1         5        10       100
model                                                                                 
BoP_All            834.0  131.558824  0.209268  0.147059  0.235294  0.382353  0.647059
```

### Data Disclaimer

The above scripts assume the presence of the following data files:

1. Knowledge Graph
    1. Format: (head, relation, tail) per row
    2. Path: *./data/knowledge_graph.csv*

2. Test Set
    1. Format: (drug, TREATS, disease) per row
    2. Path: *./data/test.csv*
3. Drug–Disease Ground Truth
    1. Format: (f"{DrugCUI}_{DiseaseCUI}", label) Label: 0 (negative) or 1 (positive)
    2. Path: *./data/test.csv*

If any of these datasets are missing, please contact bogas.ko [at] iit.demokritos.gr to obtain the required files for training and evaluation.