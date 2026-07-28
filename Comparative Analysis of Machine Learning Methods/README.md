# A Comparative Analysis of Machine Learning Methods

A comparison of four classification methods — Random Forest, Support Vector Machine (SVM), Logistic Regression, and K-Nearest Neighbours (KNN) — on a multi-class prediction task, with an accompanying written report.

## Task and data

The methods are evaluated on the [Forest Cover Type](https://archive.ics.uci.edu/dataset/31/covertype) dataset, predicting one of seven forest cover types from cartographic features (elevation, slope, soil type, distances to hydrology and roads, and others). The data is class-balanced by sampling an equal number of instances per cover type and normalised before training.

## Method

Each classifier is tuned via grid search with 10-fold cross-validation, then evaluated on a held-out test set using the macro F1 score. `Combined_code.ipynb` contains the full pipeline: preprocessing, tuning, and per-model evaluation with confusion matrices.

## Findings

Random Forest performed best (macro F1 ≈ 0.85), followed closely by KNN (≈ 0.82) and SVM (≈ 0.81), with Logistic Regression clearly weakest (≈ 0.69) — consistent with a linear model struggling on a non-linear, multi-class problem. The full methodology and results are discussed in the [project report](INST0060_Project_Report.pdf).

## Getting started

Install dependencies:
```
pip install -r requirements.txt
```

Open the notebook:
```
jupyter notebook Combined_code.ipynb
```

## Dependencies
- pandas
- NumPy
- Matplotlib
- scikit-learn