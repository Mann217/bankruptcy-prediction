# Bankruptcy Early Warning System

Predicts corporate financial distress using machine learning on financial ratio data from 6,819 Taiwan Stock Exchange companies (1999–2009).

## Overview

Built an ensemble classifier to identify insolvency risk from 95 financial ratios, optimized for recall to reflect the asymmetric cost of missed bankruptcies in credit risk screening.

## Dataset

- **Source:** Taiwan Economic Journal via [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
- **Size:** 6,819 companies | 95 financial features
- **Class distribution:** 96.8% solvent, 3.2% bankrupt (30:1 imbalance)

## Methodology

1. **Feature Selection** — Reduced 95 ratios to 13 optimal predictors via ANOVA F-value scoring and Pearson correlation pruning (threshold: 0.85)
2. **Class Imbalance** — Resolved 30:1 imbalance using SMOTE oversampling
3. **Models Trained** — Decision Tree, Random Forest, Logistic Regression, Naive Bayes, SVM, XGBoost
4. **Hyperparameter Tuning** — Bayesian optimization via BayesSearchCV
5. **Ensemble** — Voting Classifier combining all 6 models

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 93.4% |
| F1 Score | 93.6% |
| AUC-ROC | 93.4% |
| Recall | 97.3% |
| Precision | 90.2% |


## Tech Stack

`Python` `scikit-learn` `XGBoost` `imbalanced-learn` `scikit-optimize` `pandas` `seaborn`

## Files

```
BankruptcyPrediction.ipynb  — full pipeline
Bankruptcy Prediction Report.docx  — methodology report
```
