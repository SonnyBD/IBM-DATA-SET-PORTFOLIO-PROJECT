# 📊 Employee Retention Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Made with](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet)

[![View in nbviewer](https://img.shields.io/badge/View%20Notebook-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/SonnyBD/employee-retention-risk/blob/main/notebooks/Full_Retention_Model_Walkthrough.ipynb)



A machine learning–driven People Analytics project that identifies employees at risk of leaving and explains the drivers behind attrition. Built to support HR teams in making proactive, data-informed retention decisions.

![Risk Breakdown](outputs/Risk_Distribution_PieChart.png)

---

## 🧠 Objective

To build a calibrated, interpretable predictive model using HR data that estimates employee attrition risk, uncovers key retention factors, and segments employees into actionable risk tiers for HR intervention.

---

## 🧰 Tools & Techniques

- Python (pandas, scikit-learn, imbalanced-learn, SHAP)
- Random Forest Classifier + Recursive Feature Elimination (RFE)
- SMOTE for class balancing
- Probability calibration (Platt scaling)
- SHAP for model explainability
- Matplotlib & Seaborn for visualization

---

## 🔄 Workflow Summary

1. Data cleaning and feature engineering  
2. Class balancing using SMOTE  
3. Feature selection with RFE  
4. Random Forest model tuning and probability calibration  
5. Risk scoring and percentile-based tiering (Low, Moderate, High)  
6. SHAP-based interpretation of feature importance  
7. Final outputs: Excel reports, risk segmentation, and SHAP visualizations

---

## 📈 Key Results

- ✅ **88% model accuracy**
- 🎯 **98% recall** for identifying leavers (minimized false negatives)
- 🔍 **Top predictors**: Overtime, Promotion Rate, Job Satisfaction, Environment Satisfaction
- ⚠️ **10% of employees flagged as high risk** using calibrated thresholds

---

## 📁 Repository Structure

```
employee-retention-risk/
│
├── data/                           # Raw and preprocessed input data
│   ├── IBM_Test_Project_Preprocessed_Data.xlsx
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/                      # Jupyter notebook walkthrough
│   └── Full_Retention_Model_Walkthrough_UPDATED.ipynb
│
├── outputs/                        # Final visualizations & export files
│   ├── Retention_Risk_Analysis_Output.xlsx
│   └── Risk_Distribution_PieChart.png
│
├── src/                            # Modular machine learning pipeline
│   └── retention_pipeline.py
│
├── .gitignore                      # Files to exclude from version control
├── LICENSE                         # MIT open-source license
├── README.md                       # Project overview and results
└── requirements.txt                # Python environment dependencies
```

---

## 🚀 How to Run This Project

```bash
1. Clone the repository
git clone https://github.com/your-username/employee-retention-risk.git
cd employee-retention-risk

2. Install dependencies
pip install -r requirements.txt

3. Run the pipeline script
python src/retention_pipeline.py

4. Launch the notebook for a full walkthrough
jupyter notebook notebooks/Full_Retention_Model_Walkthrough_UPDATED.ipynb
```


📜 License

This project is licensed under the MIT License — open for use with attribution.
