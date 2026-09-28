Fraud Risk Classification
Arrowstack ML Internship – Project 2

1. Business Context
Imagine you’re a fraud analyst at a mobile payments company. Every day, millions of transactions flow through the system, and a tiny fraction are fraudulent. The analyst’s job is to decide which ones to block or send for manual review before money leaves the account.

The stakes are clear:

Catch fraud early → save money.

Keep false alarms low → avoid annoying legitimate customers and overloading the review team.

2. Problem in Plain Words
We need a model that looks at each transaction and says: “This looks risky” or “This looks fine.”

Input: Transaction details (type, amount, balances before/after).

Output: A fraud probability, which we turn into a yes/no decision using a threshold.

Goal: Catch most fraud (high recall) while keeping false alarms manageable (high precision).

3. Scope & Success Criteria
What’s included:

Binary fraud classification.

Logistic Regression as a baseline, Gradient Boosting as the stronger model.

Handling class imbalance, tuning thresholds, and studying precision/recall trade‑offs.

Feature importance for explainability.

What’s not included:

Real‑time deployment.

Graph/network analysis of account IDs.

Per‑transaction SHAP explanations.

Targets we’re aiming for:

Recall ≥ 0.90 (don’t miss fraud).

Precision ≥ 0.80 (don’t drown analysts in false alarms).

ROC‑AUC ≥ 0.95 (good overall separability).

4. The Dataset
We’re working with the PaySim synthetic dataset (Synthetic_Financial_datasets_log.csv):

Rows: ~6.36 million

Fraud cases: 8,213 (~0.13%)

Target column: isFraud

Key observations:

Fraud only happens in CASH_OUT and TRANSFER transactions.

In almost all fraud cases, the sender’s account is emptied (newbalanceOrig = 0).

The fraud amount usually equals the sender’s full balance.

5. Approach
Because the dataset is huge, we undersampled legitimate transactions for speed:

Keep all 8,213 fraud rows + 200,000 random legitimate rows → working set of ~208k rows (fraud rate ~3.9%).

Steps:

Feature engineering: balance‑consistency checks (errorBalanceOrig, errorBalanceDest), flags for zeroed balances.

Drop IDs and irrelevant columns. One‑hot encode transaction type.

Split into train/validation/test (70/15/15).

Train Logistic Regression (baseline).

Train Gradient Boosting (200 trees, depth 4).

Tune decision threshold using precision‑recall curve (maximize F1).

Evaluate on test set at chosen threshold and at a high‑recall setting.

Analyze errors and feature importances.

6. How to Run
Download the dataset from Kaggle.

Place the CSV in the repo root (or update the path in src/fraud.py).

Install dependencies: pip install -r requirements.txt.

Run: python src/fraud.py.

Wait a few minutes for training, then check the printed results.

Random seed is fixed → results are reproducible.

7. Validation Snapshot
Data split example:

Train: 70%

Validation: 15%

Test: 15%
