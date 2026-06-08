# Heart Failure Prediction – Random Forest, LSTM & KNN

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

**Source**: Kaggle (Heart Failure Prediction Dataset)  
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

- **Random Forest (mandatory)** – robust to outliers, handles non‑linear relationships, provides feature importance
- **LSTM** – chosen to test if a recurrent neural network can capture patterns in structured medical data (treated as a sequence of length 1)
- **KNN** – simple, non‑parametric baseline to compare against advanced methods

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

## Tech

- **Language**: Python 3.12.4
- **Libraries**: Pandas, NumPy, Scikit‑learn (RandomForestClassifier, KNeighborsClassifier, KFold, StandardScaler, LabelEncoder, confusion_matrix, roc_curve, auc), TensorFlow / Keras (Sequential, LSTM, Dense), Matplotlib, Seaborn
- **Development**: Jupyter Notebook, Google Colab
- **Version Control**: Not yet on GitHub

## Run

```bash
# Clone (after uploading to GitHub)
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction

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
