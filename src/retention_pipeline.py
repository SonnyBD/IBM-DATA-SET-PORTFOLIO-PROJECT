""" 
Retention Risk Prediction Pipeline

This script loads HR data, engineers features, trains a calibrated Random Forest model to predict employee attrition,
assigns risk tiers, and explains predictions using SHAP values. Outputs include Excel reports and visualizations.

Author: Sonny Bigras-Dewan
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

def load_and_engineer_data(filepath):
    df = pd.read_excel(filepath)

    if "JobSatisfaction" in df.columns and "WorkLifeBalance" in df.columns:
        df["Engagement_Index"] = df["JobSatisfaction"] * df["WorkLifeBalance"]
    if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
        df["Promotion_Rate"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    if "Doing_Overtime" in df.columns and "Laboratory_Technician" in df.columns:
        df["Overtime_SensitiveRole"] = df["Doing_Overtime"] * df["Laboratory_Technician"]

    df = df.drop(columns=["EmployeeNumber"], errors='ignore')
    return df

def preprocess_data(df):
    X = df.drop(columns=["Retained"])
    y = df["Retained"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, scaler

def select_features(X_scaled, y, feature_names):
    selector = RFE(RandomForestClassifier(), n_features_to_select=20)
    selector.fit(X_scaled, y)
    selected_cols = feature_names[selector.support_]
    return selected_cols

def train_model(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_resampled, y_resampled)
    rf = grid.best_estimator_

    calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
    calibrated.fit(X_resampled, y_resampled)

    return calibrated, grid.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def assign_risk(df, probs, output_path):
    df["Predicted_Prob_Stay"] = probs
    scaler = MinMaxScaler()
    df["Retention_Risk"] = 1 - scaler.fit_transform(df[["Predicted_Prob_Stay"]])

    p50 = df["Retention_Risk"].quantile(0.50)
    p90 = df["Retention_Risk"].quantile(0.90)

    df["Risk_Level"] = pd.cut(df["Retention_Risk"], bins=[-float("inf"), p50, p90, float("inf")], labels=["Low Risk", "Moderate Risk", "High Risk"])
    os.makedirs("outputs", exist_ok=True)
    df.sort_values("Retention_Risk", ascending=False).to_excel(output_path, index=False)

def plot_distribution(df, output_path):
    p50 = df["Retention_Risk"].quantile(0.50)
    p90 = df["Retention_Risk"].quantile(0.90)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Retention_Risk"], bins=50, kde=True)
    plt.axvline(p50, color='orange', linestyle='--', label='50th Percentile')
    plt.axvline(p90, color='red', linestyle='--', label='90th Percentile')
    plt.title("Retention Risk Score Distribution")
    plt.xlabel("Retention Risk")
    plt.ylabel("Number of Employees")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_shap(model, X_selected, feature_names, output_path):
    import shap
    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_model = model.calibrated_classifiers_[0].estimator
    explainer = shap.Explainer(base_model, X_selected)
    shap_values = explainer(X_selected, check_additivity=False)

    # Convert to base values and raw SHAP values
    values = shap_values.values
    feature_names = shap_values.feature_names

    # Use classic summary plot fallback (robust)
    shap.summary_plot(values, features=X_selected, feature_names=feature_names, plot_type="bar", show=False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    df = load_and_engineer_data("/Users/sonnybigras-dewan/PycharmProjects/employee-retention-risk/data/IBM_Test_Project_Preprocessed_Data.xlsx")
    X, X_scaled, y, scaler = preprocess_data(df)

    selected_cols = select_features(X_scaled, y, df.drop(columns=["Retained"]).columns)
    X_selected = df[selected_cols]
    X_scaled_selected = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_selected, y, test_size=0.2, random_state=42)

    model, best_params = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    probs = model.predict_proba(X_scaled_selected)[:, 1]
    assign_risk(df, probs, "outputs/Retention_Risk_Analysis_Output.xlsx")
    plot_distribution(df, "outputs/Retention_Risk_Distribution_Percentile.png")
    run_shap(model, X_scaled_selected, selected_cols, "outputs/SHAP_Feature_Impact.png")

    print("✔ Analysis complete. Outputs saved to /outputs.")

if __name__ == "__main__":
    main()
