import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, precision_recall_curve

# 1. Load dataset into a Pandas DataFrame using as_frame=True
cancer = load_breast_cancer(as_frame=True)
X_c, y_c = cancer.data, cancer.target

# 2. Split the data into train and test sets
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

# --- BASELINE MODEL (Balanced, standard weights) ---
lr_baseline = LogisticRegression(max_iter=10000, random_state=42)
lr_baseline.fit(X_train_c, y_train_c)
y_pred_base = lr_baseline.predict(X_test_c)

# --- COST-SENSITIVE MODEL ---
# Constraint: False Negatives (missing malignant tumors = class 0) are 10x worse.
# Heavily weight class 0 to force the model to prioritize high Recall.
lr_cost_sensitive = LogisticRegression(class_weight={0: 10.0, 1: 1.0}, max_iter=10000, random_state=42)
lr_cost_sensitive.fit(X_train_c, y_train_c)
y_pred_adj = lr_cost_sensitive.predict(X_test_c)

# 3. Quantify Performance & Error Tradeoffs
acc_base = accuracy_score(y_test_c, y_pred_base)
acc_adj = accuracy_score(y_test_c, y_pred_adj)
recall_base = recall_score(y_test_c, y_pred_base, pos_label=0)
recall_adj = recall_score(y_test_c, y_pred_adj, pos_label=0)

# Confusion matrices to check False Negatives (Row 0, Col 1 if malignant is 0)
cm_base = confusion_matrix(y_test_c, y_pred_base)
cm_adj = confusion_matrix(y_test_c, y_pred_adj)

print("--- PART 1 RESULTS FOR DEFENSE REPORT ---")
print(f"Baseline Accuracy:     {acc_base:.4f} | Malignant Recall: {recall_base:.4f}")
print(f"Cost-Sensitive Accuracy: {acc_adj:.4f} | Malignant Recall: {recall_adj:.4f}")
print(f"Accuracy Drop Accepted:  {acc_base - acc_adj:.4f}")
print(f"Baseline False Negatives (Missed Tumors):     {cm_base[0, 1]}")
print(f"Cost-Sensitive False Negatives (Missed Tumors): {cm_adj[0, 1]}")

# 4. Plot and Save the Precision-Recall Tradeoff Curve
y_scores = lr_cost_sensitive.predict_proba(X_test_c)[:, 0] # Probability of malignant class (0)
precision, recall, thresholds = precision_recall_curve(y_test_c, y_scores, pos_label=0)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='purple', lw=2, label='Cost-Sensitive PR Curve')
plt.xlabel('Recall (Malignant Detection)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Part 1: Precision-Recall Tradeoff Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('part1_pr_curve.png') # Saves the image file for your Word doc report
print("Saved part1_pr_curve.png successfully!\n")
plt.show() # Pops open the interactive graph window on your screen!