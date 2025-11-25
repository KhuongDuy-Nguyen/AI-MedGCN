# NHANES Graph Models

This project trains graph-based machine learning models to classify **diabetes vs non-diabetes** using NHANES patient data.  
Supported architectures:

- **GCN** — Graph Convolutional Network  
- **MRA** — Multi‑Relation + Attention model (inspired by MedGCN)

---

## Dataset

Download NHANES data from Kaggle:  
https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey

Place the following CSV files into the `data/` folder:

```
demographic.csv
diet.csv
examination.csv
labs.csv
medications.csv
questionnaire.csv
```

---

## Project Structure

```
project/
│
├── data/                   # NHANES CSV files
├── results/                # metrics, result tables
│
├── evaluation.py           # evaluation: ROC + confusion matrix
├── gcn_model.py            # baseline GCN model
├── nodeclassify.py         # quick prediction script
├── preparedata.py          # data loading + graph construction
│
├── train_gcn.py            # train GCN
├── train_gcn_vs_mra.py     # train + compare GCN & MRA
│
├── README.md
└── requirements.txt
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Training

Train GCN:

```
python train_gcn_nhanes.py
```

Train both models (GCN + MRA):

```
python train_nhanes_gcn_vs_mra.py
```

---

## Evaluation

Generate ROC curve, confusion matrix and classification report:

```
python evaluation_nhanes.py
```

Outputs saved in:

- `images/`
- `results/`

---

## Reference

Inspired by MedGCN: https://github.com/mocherson/MedGCN
