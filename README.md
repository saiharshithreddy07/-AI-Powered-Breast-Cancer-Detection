# 🏥 Breast Cancer Detection using Machine Learning

## 📌 About This Project

This project builds an AI-powered system to detect breast cancer from cell nucleus measurements. Using data from 569 patients, we train multiple machine learning models to classify tumors as **benign** or **malignant**.

🎯 **Goal:** Create a reliable diagnostic tool that catches every cancer case

📊 **Dataset:** Wisconsin Breast Cancer Dataset (UCI Machine Learning Repository)

🔬 **Features:** 30 cell measurements including radius, texture, perimeter, area, smoothness, compactness, and more

## ❓ Why This Matters

Breast cancer is one of the most common cancers worldwide. Early detection is crucial:

- 🟢 **Early detection:** Over 90% survival rate
- 🔴 **Late detection:** Survival rate drops significantly

Every missed diagnosis can cost a life. This project uses AI to help ensure no cancer case goes undetected.

## ⚠️ The Problem We're Solving

In medical diagnosis, there are two types of errors:

| Error Type | What Happens | Risk |
|------------|--------------|------|
| ❌ **False Negative** | Cancer patient told they're healthy | 🔴 **Dangerous** — No treatment |
| ⚠️ **False Positive** | Healthy person flagged for more tests | 🟡 Safe — Just extra testing |

**Our priority:** Maximize **Recall** (sensitivity) to catch ALL cancer cases, even if it means a few extra tests for healthy patients.

## 🤖 Machine Learning Models

We train and compare 5 different algorithms:

| Model | Description |
|-------|-------------|
| 🔬 **SVM** | Support Vector Machine — finds optimal decision boundary |
| 🌲 **Random Forest** | Ensemble of decision trees voting together |
| 🚀 **XGBoost** | Gradient boosting for high performance |
| 🧠 **Neural Network** | Deep learning approach |
| 🎯 **Ensemble** | Combines all models for best results |

## 📈 Model Performance

All models are evaluated on test data (114 patients):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SVM | 97.4% | 96.3% | 97.6% | 96.9% | 0.994 |
| Random Forest | 96.5% | 95.1% | 97.6% | 96.3% | 0.992 |
| XGBoost | 96.5% | 95.1% | 97.6% | 96.3% | 0.993 |
| Neural Network | 95.6% | 94.0% | 95.2% | 94.6% | 0.987 |
| **Ensemble** | **97.4%** | **96.3%** | **97.6%** | **96.9%** | **0.995** |

### 📊 Key Metrics Explained

- **Accuracy:** Overall correct predictions
- **Precision:** When we predict cancer, how often are we right?
- **Recall:** Of all actual cancer cases, how many did we catch? ⭐ *Our priority*
- **F1-Score:** Balance between precision and recall
- **ROC-AUC:** Overall model quality (1.0 = perfect)

## ✅ Results

- 🎯 **97.6% Recall** — Catches nearly all malignant cases
- 📊 **97.4% Accuracy** — Highly reliable predictions
- 🏆 **0.995 ROC-AUC** — Excellent discrimination between classes

The ensemble model combines predictions from all 5 models to achieve the best overall performance.

## 🔍 Explainability

We use **SHAP (SHapley Additive exPlanations)** to understand which features drive each prediction. This makes our AI transparent and trustworthy — doctors can see *why* the model made its decision.

Top predictive features:
- Worst Perimeter
- Worst Concave Points
- Worst Radius
- Mean Concave Points
- Worst Area

## 🛠️ Technology Stack

- 🐍 Python 3.8+
- 📊 Pandas, NumPy
- 🤖 Scikit-learn, XGBoost
- 📈 Matplotlib, Seaborn
- 🔍 SHAP
- 📓 Jupyter Notebook

## 📁 Project Structure

```
├── AI_Medical_Diagnostic_Tool_(Breast_Cancer_Detection).ipynb
├── README.md
├── roc_curves_comparison.png
├── metrics_comparison.png
├── shap_feature_importance.png
└── shap_waterfall.png
```

## 🚀 How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn shap

# Open the notebook
jupyter notebook AI_Medical_Diagnostic_Tool_(Breast_Cancer_Detection).ipynb
```

---

⚕️ *This is an educational project demonstrating machine learning in healthcare. It is NOT intended for clinical diagnosis. Always consult qualified medical professionals for health concerns.*
