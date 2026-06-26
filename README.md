# Drug Consumption — Cannabis Use Prediction

An end-to-end machine learning project that predicts cannabis addiction level based on demographic and personality trait data.

---

## Problem Statement

Using survey data on drug consumption patterns, this project builds a **multi-class classifier** to predict whether an individual is:

- **0 — Not addicted** (never used, or used over a decade ago)
- **1 — Mildly addicted** (used in the last year or last month)
- **2 — Addicted** (used in the last week or last day)

---

## Dataset

| File | Description |
|------|-------------|
| `Drug_Consumption.csv` | Raw survey data with demographic and personality features, plus drug usage frequency for 32 substances |
| `Drug_Consumption_Quantified.csv` | Pre-quantified version of the dataset |

### Features

**Demographic:**
- `Age`, `Gender`, `Education`, `Country`, `Ethnicity`

**Big Five Personality Scores (NEO-FFI-R):**
- `Nscore` (Neuroticism), `Escore` (Extraversion), `Oscore` (Openness), `AScore` (Agreeableness), `Cscore` (Conscientiousness)

**Impulsivity Measures:**
- `Impulsive`, `SS` (Sensation Seeking)

**Engineered Features:**
- `Nicotine_user`, `Coke_user`, `Ecstasy_user`, `Alcohol_user` — binary flags derived from usage frequency of correlated substances

**Target:** `Cannabis` usage frequency, mapped to 3 classes (0, 1, 2)

---

## Project Workflow

### 1. Data Cleaning
- Dropped the `ID` column
- Mapped `Age` ranges to midpoint numeric values
- Consolidated rare categories in `Country` and `Ethnicity`

### 2. Feature Engineering
- Created binary flags for 4 correlated drug usages (Nicotine, Coke, Ecstasy, Alcohol)
- Encoded `Education` ordinally (0–8)
- Binary-encoded `Gender`

### 3. Target Encoding
Cannabis usage classes (`CL0`–`CL6`) were mapped to 3 addiction levels:
```
CL0, CL1, CL2  →  0  (Not addicted)
CL3, CL4        →  1  (Mildly addicted)
CL5, CL6        →  2  (Addicted)
```

### 4. Exploratory Data Analysis
- Correlation heatmap across all numeric features
- Distribution plots for all features
- Class balance analysis
- Countplots of Gender, Education, Country, Ethnicity vs label
- Box plots of Age and personality scores vs label

### 5. Preprocessing
- `OneHotEncoder` for remaining categorical features
- `StandardScaler` for numerical features
- Combined via `ColumnTransformer`

### 6. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on the training set

### 7. Model Training & Evaluation
Nine models were trained and compared:

| Model | Notes |
|-------|-------|
| Logistic Regression | `class_weight='balanced'` |
| SVM | RBF kernel, `probability=True` |
| K-Nearest Neighbors | — |
| Naive Bayes (Gaussian) | — |
| Decision Tree | `max_depth=5` |
| Random Forest | 100 estimators, `max_depth=5` |
| AdaBoost | — |
| Gradient Boosting | — |
| XGBoost | `max_depth=3`, `learning_rate=0.1` |

Metrics tracked: Accuracy, F1 (macro), Precision, Recall, ROC-AUC, Overfitting Gap

### 8. Hyperparameter Tuning
`RandomizedSearchCV` was used to tune **SVM** and **Logistic Regression**, with 3-fold cross-validation and 100 iterations.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

---

## Usage

1. Clone the repository and navigate to the project folder.
2. Ensure both CSV files are in the same directory as the notebook.
3. Open and run `end_to_end_project.ipynb` in Jupyter Notebook or JupyterLab.

```bash
jupyter notebook end_to_end_project.ipynb
```

---

## Project Structure

```
First project/
├── end_to_end_project.ipynb        # Main notebook
├── Drug_Consumption.csv            # Raw dataset
└── Drug_Consumption_Quantified.csv # Pre-quantified dataset
```

---

## Key Design Decisions

- **Dropped personality scores** `Escore`, `AScore`, `Nscore` after EDA showed low discriminative power for cannabis use.
- **Stratified train/test split** (`stratify=y`) to preserve class proportions.
- **SMOTE applied only on training data** to prevent data leakage.
- **Multi-class ROC-AUC** computed using `ovr` (One-vs-Rest) strategy.