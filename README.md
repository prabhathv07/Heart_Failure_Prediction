# Heart Failure Prediction – Random Forest, LSTM & KNN

![Python](https://img.shields.io/badge/Python-3.12.4-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Developed and compared three binary classifiers (Random Forest, LSTM, KNN) to predict heart disease from patient clinical data, achieving 86.8% accuracy with Random Forest using 10‑fold cross‑validation.

## Problem

Cardiovascular diseases are the leading cause of death worldwide. Early and accurate prediction of heart disease from routine clinical parameters can save lives by enabling preventive care. The challenge is to build a reliable predictive model that works on real‑world, imbalanced medical data, and to compare traditional machine learning, ensemble methods, and deep learning on the same task.

## Architecture

```
Raw Data (918 patients, 12 features)
         │
         ▼
Data Preprocessing
  - Label encoding for categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
  - StandardScaler for numerical features
         │
         ▼
10‑Fold Cross‑Validation (shuffled, random_state=42)
         │
         ├── Fold 1: Train (826) → Test (92)
         ├── Fold 2: Train (826) → Test (92)
         ├── ...
         └── Fold 10: Train (827) → Test (91)
         │
         ▼
Three Parallel Models
  ├── Random Forest (ensemble of decision trees)
  ├── LSTM (long short‑term memory neural network)
  └── KNN (k‑nearest neighbors)
         │
         ▼
Per‑Fold & Average Metrics
  - TP, TN, FP, FN
  - Accuracy, Balanced Accuracy, Precision, Recall, F1
  - TSS, HSS, Brier Score (BS), Brier Skill Score (BSS)
  - ROC / AUC
```

## Approach

### Dataset

**Source**: [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)  
**Size**: 918 records, 12 features, binary target (HeartDisease: 0 = no disease, 1 = disease)  
**Distribution**: 508 positive (55.3%), 410 negative (44.7%) – moderately balanced  
**Features**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

### Model / Pipeline

| Step | Description |
|------|-------------|
| Preprocessing | LabelEncoder for 5 categorical columns; StandardScaler for numerical columns |
| Validation | 10‑fold cross‑validation (shuffle = True, random_state = 42) – all metrics computed manually per fold |
| Random Forest | RandomForestClassifier() with default hyperparameters (scikit‑learn) |
| LSTM | Sequential model: LSTM(50) → Dense(1, sigmoid); Adam optimizer; binary cross‑entropy; 50 epochs, batch_size=32 |
| KNN | KNeighborsClassifier() with default k (scikit‑learn) |
| Metrics | All computed manually from confusion matrices; ROC/AUC using roc_curve and auc; BS/BSS using custom formulas |

### Why these algorithms?

- **Random Forest (mandatory)** – robust to outliers, handles non‑linear relationships, provides feature importance, and is well-suited for tabular medical data with mixed feature types
- **LSTM** – while typically used for sequential data, this experiment tests whether treating tabular features as a sequence can capture complex feature interactions that traditional methods might miss; serves as a deep learning baseline to compare against tree-based methods
- **KNN** – simple, non‑parametric baseline to compare against advanced methods; provides intuition about local similarity patterns in the feature space

## Results (numbers)

### Average Performance across 10 folds

| Metric | Random Forest | LSTM | KNN |
|--------|---------------|------|-----|
| Accuracy | 0.868 | 0.842 | 0.710 |
| Balanced Accuracy | 0.866 | 0.843 | 0.705 |
| Precision | 0.864 | 0.862 | 0.726 |
| Recall (TPR) | 0.907 | 0.851 | 0.759 |
| F1 Score | 0.883 | 0.855 | 0.741 |
| TSS | 0.732 | 0.686 | 0.410 |
| HSS | 0.731 | 0.681 | 0.410 |
| Brier Score (BS) | 0.103 | 0.125 | 0.213 |
| Brier Skill Score (BSS) | 0.602 | 0.470 | 0.116 |
| AUC | ~0.94 | ~0.90 | ~0.74 |

### Sample fold results (Random Forest – Fold 1)

- TP = 50, TN = 33, FP = 5, FN = 4
- TPR = 0.926, TNR = 0.868, FPR = 0.132, FNR = 0.074
- Accuracy = 0.902, Balanced Accuracy = 0.897
- Precision = 0.909, Recall = 0.926, F1 = 0.917
- TSS = 0.794, HSS = 0.797, BS = 0.097, AUC = 0.946

### Random Forest – Best Performing Algorithm

- **Accuracy**: 86.8% over 10 folds
- **F1 Score**: 0.883 – excellent balance of precision and recall
- **AUC ≈ 0.94** – outstanding discriminative power
- **Low Brier Score (0.103)** – well‑calibrated probabilities

## Clinical Insights & Key Findings

### Model Performance Interpretation

The Random Forest model achieved **86.8% accuracy** with a **high recall of 90.7%**, meaning it correctly identifies 9 out of 10 patients who actually have heart disease. This is clinically significant because:
- **False negatives are minimized** – missing a heart disease diagnosis is far more dangerous than a false positive
- **High TSS (0.732)** indicates strong discriminative ability beyond random chance
- **Well-calibrated probabilities** (Brier Score 0.103) suggest the model's confidence scores are reliable for clinical decision-making

### Feature Importance Analysis

Based on the Random Forest model's feature importance, the most predictive clinical indicators for heart disease are:

1. **ST_Slope** – The slope of the peak exercise ST segment is the strongest predictor, with flat and downward slopes strongly associated with heart disease
2. **ChestPainType** – Asymptomatic (ASY) chest pain shows high correlation with heart disease, highlighting the danger of silent heart attacks
3. **MaxHR** – Lower maximum heart rate during exercise is associated with higher heart disease risk
4. **Oldpeak** – ST depression during exercise (Oldpeak) is a significant predictor
5. **Age** – Older patients show increased risk, as expected in cardiovascular disease

### Clinical Implications

- **Silent ischemia detection**: The model's strong reliance on asymptomatic chest pain patterns suggests it can help identify patients with silent heart disease who might otherwise be missed
- **Exercise stress test value**: ST_Slope and Oldpeak being top predictors validates the clinical importance of exercise stress testing
- **Risk stratification**: The model can be used as a decision support tool to flag high-risk patients for further cardiac evaluation

## Visual Outputs

The notebook includes the following visualizations:

- **ROC Curves**: Comparison of all three models showing Random Forest's superior AUC (0.94)
- **Confusion Matrices**: Per-fold confusion matrices showing TP, TN, FP, FN distributions
- **Correlation Heatmap**: Feature correlation matrix showing relationships between clinical parameters
- **Feature Distribution Histograms**: Distribution of each feature across the dataset
- **Pair Plots**: Scatter plot matrix colored by heart disease status

## Tech

- **Language**: Python 3.12.4
- **Libraries**: Pandas, NumPy, Scikit‑learn (RandomForestClassifier, KNeighborsClassifier, KFold, StandardScaler, LabelEncoder, confusion_matrix, roc_curve, auc), TensorFlow / Keras (Sequential, LSTM, Dense), Matplotlib, Seaborn
- **Development**: Jupyter Notebook
- **Repository**: https://github.com/prabhathv07/Heart_Failure_Prediction

## Run

```bash
# Clone the repository
git clone https://github.com/prabhathv07/Heart_Failure_Prediction.git
cd Heart_Failure_Prediction

# Install dependencies
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

# Run the Jupyter notebook
jupyter notebook heart_failure_prediction.ipynb
```

The program will:
- Load heart.csv
- Preprocess data (encode categoricals, scale)
- Perform 10‑fold cross‑validation on all three models
- Print per‑fold and average metrics
- Display ROC curves and comparison tables

## What I'd do next

- **Hyperparameter tuning** – Use GridSearchCV for Random Forest (n_estimators, max_depth) and KNN (k, weights); tune LSTM layers, units, dropout, and epochs.
- **Feature engineering** – Create interaction features (e.g., age × cholesterol, chest pain type × exercise angina) to improve recall.
- **Explainability** – Generate SHAP or LIME explanations for Random Forest predictions to help clinicians trust the model.
- **Deployment** – Wrap the best model (Random Forest) in a FastAPI or Flask REST API and deploy on AWS/GCP.
- **Handle missing values** – The current dataset has no missing values, but real clinical data does – implement imputation strategies.
- **Class imbalance** – Since the dataset is only mildly imbalanced, SMOTE didn't help much, but for larger imbalance I would test SMOTE + ENN.
