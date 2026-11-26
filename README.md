# Heart Disease Prediction and Patient Risk Analysis

A Machine Learning project that predicts the presence of heart disease and explores patient risk patterns using the UCI Heart Disease dataset.

## Project Overview

This project analyzes patient health information and applies machine learning techniques to predict whether a patient may have heart disease.

**Dataset:** UCI Heart Disease Dataset  
**Total Patients:** 303  
**Original Features:** 13  
**Target:** Heart Disease (0 = No, 1 = Yes)

## Objectives

- Analyze patient health data.
- Clean missing values and handle outliers.
- Perform exploratory data analysis (EDA).
- Train and compare machine learning models.
- Optimize model parameters using cross-validation.
- Predict heart disease using patient information.
- Explore patient risk groups using K-Means clustering.
- Provide an interactive Streamlit application.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## Machine Learning Algorithms

1. Linear Regression
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. K-Means Clustering
5. Random Forest (model optimization and comparison)

## Model Performance

The following are the actual test results obtained in this project:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.25% | 82.76% | 85.71% | 84.21% |
| KNN | **88.52%** | **88.89%** | 85.71% | **87.27%** |
| Random Forest | 86.89% | 81.25% | **92.86%** | 86.67% |

**Selected Model:** K-Nearest Neighbors (KNN), based on the highest test F1 Score.

Random Forest achieved the highest recall. The model selection is for this educational project and does not establish clinical suitability.

## Project Structure

```text
Heart-Disease-Prediction/
│
├── dataset/
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.csv
│   └── ...
│
├── models/
│   ├── scaler.pkl
│   ├── best_heart_disease_model.pkl
│   └── ...
│
├── reports/
│   ├── statistical_summary.csv
│   ├── eda_summary.csv
│   ├── model_optimization_results.csv
│   ├── best_model_summary.csv
│   └── ...
│
├── src/
│   ├── download_dataset.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── logistic_regression.py
│   ├── knn_model.py
│   ├── kmeans_clustering.py
│   ├── model_optimization.py
│   ├── model_comparison.py
│   └── ...
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/tastybites-project/Heart-Disease-Prediction.git
```

### 2. Open the project folder

```bash
cd Heart-Disease-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset Source

UCI Machine Learning Repository — Heart Disease Dataset:

https://archive.ics.uci.edu/dataset/45/heart+disease

## Disclaimer

This project is developed for educational and academic purposes only. Its predictions, risk scores, and risk categories are not medical diagnoses and must not be used for clinical decisions. Always consult qualified healthcare professionals for medical advice.

## Author

**Animesh Manna**

GitHub: https://github.com/tastybites-project