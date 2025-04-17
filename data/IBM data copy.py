# Employee Retention Risk Analysis with Feature Optimization and Probability Calibration

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Step 1: Load data
df = pd.read_excel("IBM_Test_Project_Preprocessed_Data.xlsx")

# Step 2: Create engineered features
if "JobSatisfaction" in df.columns and "WorkLifeBalance" in df.columns:
    df["Engagement_Index"] = df["JobSatisfaction"] * df["WorkLifeBalance"]
if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
    df["Promotion_Rate"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
if "Doing_Overtime" in df.columns and "Laboratory_Technician" in df.columns:
    df["Overtime_SensitiveRole"] = df["Doing_Overtime"] * df["Laboratory_Technician"]

# Step 3: Prepare features and target
df = df.drop(columns=["EmployeeNumber"], errors='ignore')
X = df.drop(columns=["Retained"])
y = df["Retained"]

# Step 4: Scale features
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Apply SMOTE
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Step 7: Recursive Feature Elimination (RFE)
rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
rfe_selector.fit(X_scaled, y)
X_selected = X.loc[:, rfe_selector.support_]
X_scaled_selected = scaler_std.fit_transform(X_selected)
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_scaled_selected, y, test_size=0.2, random_state=42)
X_train_sel_res, y_train_sel_res = sm.fit_resample(X_train_sel, y_train_sel)

# Step 8: Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_sel_res, y_train_sel_res)
rf_model = rf_grid.best_estimator_

# Step 9: Calibrate probabilities
calibrated_rf = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
calibrated_rf.fit(X_train_sel_res, y_train_sel_res)

# Step 10: Evaluate calibrated model
y_pred_rf = calibrated_rf.predict(X_test_sel)
print("Random Forest Best Parameters:", rf_grid.best_params_)
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test_sel, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test_sel, y_pred_rf))
print("Random Forest Accuracy:", accuracy_score(y_test_sel, y_pred_rf))

# Step 11: Cross-validation check
cv_scores = cross_val_score(calibrated_rf, X_scaled_selected, y, cv=5, scoring='f1')
print("Cross-validated F1 scores:", cv_scores)

# Step 12: Predict probabilities and calculate retention risk
df["Predicted_Prob_Stay"] = calibrated_rf.predict_proba(X_scaled_selected)[:, 1]
scaler = MinMaxScaler()
df["Retention_Risk"] = 1 - scaler.fit_transform(df[["Predicted_Prob_Stay"]])

# Step 13: Define risk levels using percentiles
percentile_90 = df["Retention_Risk"].quantile(0.90)
percentile_50 = df["Retention_Risk"].quantile(0.50)

df["Risk_Level"] = pd.cut(
    df["Retention_Risk"],
    bins=[-float("inf"), percentile_50, percentile_90, float("inf")],
    labels=["Low Risk", "Moderate Risk", "High Risk"]
)

# Step 14: Export results
df_sorted = df.sort_values(by="Retention_Risk", ascending=False)
df_sorted.to_excel("Retention_Risk_Analysis_Output.xlsx", index=False)

# Step 15: Plot risk distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Retention_Risk"], bins=50, kde=True)
plt.axvline(percentile_50, color='orange', linestyle='--', label='50th Percentile')
plt.axvline(percentile_90, color='red', linestyle='--', label='90th Percentile')
plt.title("Retention Risk Score Distribution with Percentile Thresholds")
plt.xlabel("Retention Risk")
plt.ylabel("Number of Employees")
plt.legend()
plt.tight_layout()
plt.savefig("Retention_Risk_Distribution_Percentile.png")
plt.show()

# Step 16: SHAP Explainability using rf_model (disable additivity check to prevent error)
explainer = shap.Explainer(rf_model, X_scaled_selected)
shap_values = explainer(X_scaled_selected, check_additivity=False)
shap.summary_plot(shap_values, X_selected, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("SHAP_Feature_Impact.png")
plt.show()

print("Analysis complete. Outputs saved to 'Retention_Risk_Analysis_Output.xlsx' and PNG charts.")
