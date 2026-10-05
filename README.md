# Disease Prediction from Medical Data

## Project Overview

This project uses Machine Learning to predict whether a breast tumor is likely to be malignant or benign based on medical diagnostic measurements.

The project was developed as part of the CodeAlpha Machine Learning Internship.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset available through Scikit-learn.

- Samples: 569
- Features: 30
- Classes: Malignant and Benign

## Machine Learning Algorithms

Three classification algorithms were implemented and compared:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| Random Forest | 95.61% |
| SVM | 98.25% |

Logistic Regression was selected as the final model.

## Data Preprocessing

The following preprocessing steps were performed:

- Dataset loading
- Train-test split
- Feature standardization using StandardScaler
- Model training
- Model evaluation

## Evaluation

The Logistic Regression model achieved an accuracy of **98.25%** on the test dataset.

Confusion matrix:

```text
[[41  1]
 [ 1 71]]