# Employee Retention Risk Prediction

A machine learning-based project to identify employees at risk of leaving and help HR make data-driven retention decisions.

## 🧠 Objective
To build a predictive model for employee retention using historical HR data, identify risk drivers, and present insights for intervention.

## 🧰 Tools Used
- Python (pandas, scikit-learn, shap, imblearn)
- Random Forest + RFE + SMOTE
- Probability calibration
- SHAP for model explainability
- Matplotlib / Seaborn for visualization

## 🔍 Workflow Summary
1. Data preprocessing & feature engineering
2. Feature selection via Recursive Feature Elimination (RFE)
3. Model training with cross-validation (Random Forest)
4. Calibration of probabilities for retention risk scoring
5. Risk tiering (Low, Moderate, High) using percentile thresholds
6. SHAP-based model explainability
7. Output: Excel + visual reports

## 📈 Key Results
- 88% model accuracy
- 98% recall for identifying leavers
- 10% of employees flagged as high risk
- SHAP revealed top predictors: Overtime, Promotion Rate, Job Satisfaction

## 📦 Outputs
- `Retention_Risk_Analysis_Output.xlsx`
- `Feature_Importance_RF.png`
- `SHAP_Feature_Impact.png`
- `Retention_Risk_Distribution_Percentile.png`

## 📂 Repository Structure
```
employee-retention-risk/
│
├── data/                      # Input dataset
├── notebooks/                 # Optional: Jupyter notebooks
├── outputs/                   # Visuals and Excel outputs
├── src/                       # Python scripts
├── README.md                  # Project overview
├── requirements.txt           # Dependency list
```

## 📜 License
This project is for educational and demonstration purposes.
