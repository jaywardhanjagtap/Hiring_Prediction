# 💼 Candidate Hiring Prediction & Data Leakage Auditing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline designed to predict candidate hiring decisions from resume profiles (experience, technical skills, certifications, education, and portfolio projects). 

Beyond standard predictive modeling, this project highlights a critical machine learning concept: **identifying, auditing, and mitigating target data leakage** caused by downstream proxy scores.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Engineering Insights](#-key-engineering-insights)
- [Dataset Summary](#-dataset-summary)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Evaluation & Benchmarking](#-model-evaluation--benchmarking)
- [Feature Importance](#-feature-importance)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🔍 Project Overview

Recruitment teams receive hundreds of applications per role. Automating preliminary candidate evaluations requires models that make decisions based on genuine candidate competencies rather than artifactual or proxy variables.

This project delivers:
1. **Multi-Label Skill Extraction:** Parsing and vectorizing unstructured candidate skill lists into distinct binary indicators.
2. **Target Leakage Auditing:** Demonstrating how the `AI Score` variable leaks ground-truth target data and properly removing it to ensure genuine generalization.
3. **Multi-Model Comparison:** Evaluating linear, distance-based, and rule-based algorithms with standard scaling benchmarks.
4. **Hyperparameter Tuning:** Fine-tuning regularization parameters via cross-validated Grid Search.
5. **Model Interpretability:** Extracting feature importances to reveal what hiring managers prioritize most.

---

## ⚠️ Key Engineering Highlight: Detecting & Preventing Data Leakage

In early exploratory data analysis, the variable `AI Score (0-100)` exhibited a correlation of **0.84** with `Recruiter Decision`, with hired candidates almost deterministically having higher AI scores. 

* **The Trap:** Leaving `AI Score` in the feature matrix yields a trivial ~100% accuracy model, but the model learns nothing about actual candidate qualifications—it merely mirrors a downstream automated score.
* **The Solution:** The pipeline intentionally isolates and drops `AI Score` prior to training.
* **The Experiment:** A dedicated proof-of-concept section explicitly trains a model with `AI Score` included to document and visualize the synthetic performance distortion caused by target leakage.

---

## 📊 Dataset Summary

The dataset comprises **1,000 applicant records** across multiple tech roles:

| Column Name | Type | Description |
|---|---|---|
| `Resume_ID` | Identifier | Unique applicant ID (dropped during modeling) |
| `Name` | Text | Candidate name (dropped during modeling) |
| `Skills` | Text | Comma-separated technical competencies |
| `Experience (Years)` | Integer | Total years of professional experience (0–10) |
| `Education` | Categorical | Highest degree (B.Sc, B.Tech, M.Tech, MBA, PhD) |
| `Certifications` | Categorical | Certifications (AWS Certified, Google ML, Deep Learning Specialization, None) |
| `Job Role` | Categorical | Target role (AI Researcher, Data Scientist, Cybersecurity Analyst, Software Engineer) |
| `Salary Expectation ($)` | Numeric | Annual compensation request |
| `Projects Count` | Integer | Number of completed portfolio projects (0–10) |
| `AI Score (0-100)` | Numeric | Automated resume score (audited & isolated) |
| `Recruiter Decision` | Binary (Target) | Hiring outcome (`Hire` vs. `Reject`) |

---

## ⚙️ Machine Learning Workflow

```
[ hiring_data.csv ]
         │
         ▼
[ Exploratory Data Analysis & Leakage Audit ]
         │
         ▼
[ Feature Engineering ]
  ├── Skill parsing & Multi-Hot vectorization (14 unique skills)
  ├── Skills Count aggregation
  └── One-Hot Encoding (Education, Certifications, Job Roles)
         │
         ▼
[ Train / Test Stratified Split (80 / 20) ] ── (AI Score excluded)
         │
         ▼
[ StandardScaler Normalization ]
         │
         ▼
[ Model Benchmarking & GridSearchCV ]
  ├── Logistic Regression (Scaled vs. Unscaled)
  ├── K-Nearest Neighbors (KNN)
  └── Decision Tree Classifier
         │
         ▼
[ Evaluation & Feature Importance Analysis ]
```

---

## 📈 Model Evaluation & Benchmarking

Models were trained and evaluated on a held-out test set (200 records, stratified):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Tuned)** | **1.000** | **1.000** | **1.000** | **1.000** |
| Logistic Regression (Scaled) | 0.985 | 0.994 | 0.988 | 0.991 |
| Logistic Regression (Unscaled) | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree | 0.935 | 0.963 | 0.957 | 0.960 |
| K-Nearest Neighbors (k=5) | 0.830 | 0.848 | 0.963 | 0.902 |

*Best hyperparameters for Logistic Regression:* `C: 100`, `penalty: 'l2'`, `solver: 'lbfgs'`.

---

## 🏆 Feature Importance

Feature importance extracted from model coefficients reveals the primary decision drivers:

1. **Experience (Years):** Primary predictor. Candidates with 5+ years experience showed near-100% acceptance rates.
2. **Projects Count:** Secondary key factor. High project completion demonstrates practical competency.
3. **Certifications (None):** Negative weight; lacking certifications notably reduces interview conversion.
4. **Skills Count & Specific Technologies:** Core competencies (e.g., `TensorFlow`, `React`, `Python`) provided consistent positive signal.

---

## 📂 Repository Structure

```
HiringProject/
│
├── hiring_data.csv            # Dataset containing candidate profiles
├── hiring_prediction.ipynb    # Documented analysis & ML pipeline notebook
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files and folders to exclude from version control
└── README.md                  # Project overview & documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/HiringProject.git
cd HiringProject
```

### 2. Set up a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook hiring_prediction.ipynb
```

---

## 👤 Author
- **Jaywardhan Jagtap** - [GitHub Profile](https://github.com/jaywardhanjagtap)
