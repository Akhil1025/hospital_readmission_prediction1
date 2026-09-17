# Case Study 1 — Hospital Readmission Prediction

## Objective

Predict whether a patient will be **readmitted within 30 days** using hospital/patient record features and **Logistic Regression with L2 regularization**.

The model is evaluated using **ROC-AUC**, and the effect of false positives and false negatives is discussed.

## Dataset

This project is designed for the **Diabetes 130-US hospitals for years 1999-2008** dataset.

Download the dataset and place:

```text
diabetic_data.csv
```

inside:

```text
data/
```

The final structure should be:

```text
case_study_1_hospital_readmission/
├── case_study_1.py
├── requirements.txt
├── README.md
└── data/
    └── diabetic_data.csv
```

Do **not** commit the large dataset to GitHub unless your course/instructor permits it.

## Features used

The implementation uses examples of:

- Diagnosis codes (`diag_1`, `diag_2`, `diag_3`)
- Age
- Time spent in hospital
- Number of lab procedures
- Number of procedures
- Number of medications
- Previous outpatient visits
- Previous emergency visits
- Previous inpatient visits
- Number of diagnoses
- Admission/discharge information
- Diabetes medication/change indicators

The target is:

```text
readmitted == "<30" → 1
otherwise            → 0
```

## Why Logistic Regression?

Logistic regression predicts a probability between 0 and 1 and is suitable for binary classification.

The model uses:

```python
LogisticRegression(
    penalty="l2",
    solver="liblinear",
    C=1.0
)
```

L2 regularization penalizes excessively large coefficients and can help reduce overfitting.

## Run

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python case_study_1.py
```

The program produces:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- Comparison of thresholds 0.50 and 0.30
- False-positive / false-negative clinical cost discussion

## Threshold discussion

A default threshold of `0.50` is not automatically the correct clinical threshold.

If false negatives are considered more costly, a lower threshold such as `0.30` can flag more potentially high-risk patients. This generally increases recall but also increases false positives.

The threshold should ultimately be chosen using the clinical context and the relative cost of each type of error.

## Important note

This is an educational machine-learning case study, not a clinically validated prediction system. Real clinical deployment would require external validation, careful feature governance, privacy protections, calibration, fairness evaluation, and clinical oversight.
