import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

np.random.seed(42)

df = pd.read_csv("Synthetic_Financial_datasets_log.csv")

fraud = df[df["isFraud"] == 1]
non_fraud = df[df["isFraud"] == 0].sample(n=200000, random_state=42)
df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["origBalanceZeroed"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
df["destBalanceZeroBefore"] = (df["oldbalanceDest"] == 0).astype(int)

df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])
df = pd.get_dummies(df, columns=["type"], drop_first=True)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(X_train_scaled, y_train)
baseline_val_proba = baseline.predict_proba(X_val_scaled)[:, 1]

gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb.fit(X_train, y_train)
gb_val_proba = gb.predict_proba(X_val)[:, 1]

def evaluate(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

print("=== VALIDATION SET COMPARISON (threshold 0.5) ===")
val_results = pd.DataFrame([
    evaluate("Logistic Regression (baseline)", y_val, (baseline_val_proba >= 0.5).astype(int), baseline_val_proba),
    evaluate("Gradient Boosting", y_val, (gb_val_proba >= 0.5).astype(int), gb_val_proba)
])
print(val_results.to_string(index=False))

precisions, recalls, thresholds = precision_recall_curve(y_val, gb_val_proba)
threshold_table = pd.DataFrame({"threshold": thresholds, "precision": precisions[:-1], "recall": recalls[:-1]})
threshold_table["f1"] = 2 * threshold_table.precision * threshold_table.recall / (threshold_table.precision + threshold_table.recall + 1e-9)
best_row = threshold_table.loc[threshold_table.f1.idxmax()]
print("\n=== BEST F1 THRESHOLD (validation set) ===")
print(best_row)

chosen_threshold = best_row.threshold
gb_test_proba = gb.predict_proba(X_test)[:, 1]
gb_test_pred = (gb_test_proba >= chosen_threshold).astype(int)

print(f"\n=== TEST SET PERFORMANCE (Gradient Boosting, threshold={chosen_threshold:.3f}) ===")
print(evaluate("Gradient Boosting (test)", y_test, gb_test_pred, gb_test_proba))
print(classification_report(y_test, gb_test_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, gb_test_pred))

high_recall_threshold = threshold_table[threshold_table.recall >= 0.9].threshold.max()
high_recall_pred = (gb_test_proba >= high_recall_threshold).astype(int)
print(f"\n=== HIGH-RECALL OPERATING POINT (threshold={high_recall_threshold:.3f}) ===")
print(evaluate("Gradient Boosting (high recall)", y_test, high_recall_pred, gb_test_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, high_recall_pred))

importances = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== FEATURE IMPORTANCES ===")
print(importances)

errors = X_test.copy()
errors["actual"] = y_test.values
errors["predicted"] = gb_test_pred
errors["proba"] = gb_test_proba
missed_fraud = errors[(errors.actual == 1) & (errors.predicted == 0)]
false_alarms = errors[(errors.actual == 0) & (errors.predicted == 1)]
print(f"\nMissed fraud: {len(missed_fraud)}, False alarms: {len(false_alarms)}")
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print(f"Fraud rate in working set: {y.mean():.5f}")

model_card = {
    "model_type": "GradientBoostingClassifier",
    "imbalance_handling": "class_weight=balanced (baseline) + threshold tuning (final model); undersampled non-fraud to 200k rows, kept all fraud rows",
    "chosen_threshold": float(chosen_threshold),
    "test_roc_auc": roc_auc_score(y_test, gb_test_proba),
    "top_features": importances.head(5).to_dict()
}
print("\n=== MODEL CARD ===")
print(model_card)