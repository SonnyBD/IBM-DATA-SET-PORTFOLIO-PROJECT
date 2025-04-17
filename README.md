# Employee Retention Risk Prediction
A machine learning-based project to identify employees at risk of leaving and help HR make data-driven retention decisions.

![Retention Risk Summary](outputs/Retention_Risk_Analysis_Output.pdf)

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

## 📂 Repository Structure
```
employee-retention-risk/
│
├── data/                      # Input dataset
├── notebooks/                 # Jupyter notebooks
├── outputs/                   # Visuals and Excel outputs
├── src/                       # Python scripts
├── README.md                  # Project overview
├── requirements.txt           # Dependency list
```

## 🚀 How to Run This Project

1. Clone the repo  
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python src/retention_model.py
   ```
4. Or explore in Jupyter:
   ```bash
   jupyter notebook notebooks/Retention_Model_Development.ipynb
   ```

---

## 💼 Why This Project Matters

This project simulates a real-world People Analytics workflow where HR can identify employees at high risk of leaving. Using explainable AI (SHAP), the model becomes transparent and actionable — enabling HR to prioritize interventions based on data.

---

## 📜 License

This project is for educational and portfolio use. Feel free to use the structure or approach with credit.
