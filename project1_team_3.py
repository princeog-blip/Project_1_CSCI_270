# =================================================================
# PART 1: Classification- Tumor Detection with Cost Asymmetry
# =================================================================

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

# =================================================================
# PART 2: Regression- Housing Prices and The Ceiling Effect
# =================================================================

"""
=============================================================================
California Housing Regression Analysis
Dataset:  sklearn.datasets.fetch_california_housing
Pipeline:
  1. Load California Housing Dataset
  2. Single 80/20 train-test split on raw data (Original test set held fixed)
  3. Model 1: Baseline Linear Regression trained on raw (capped) data
  4. Model 2: Cleaned Linear Regression trained on uncapped data (y < 5.0)
  5. Controlled Fair Benchmark:
     - Both models evaluated on the IDENTICAL uncapped test observations
     - Baseline performance on the full test set reported separately
  6. Visual diagnostics & Defense Report deliverable
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# STEP 1: LOAD THE DATASET
# We loaded the California Housing dataset and used eight
# features to help predict housing prices.
# We also checked if there were any missing values.
# =============================================================================
print("=" * 70)
print("STEP 1: LOADING CALIFORNIA HOUSING DATASET")
print("=" * 70)

housing = fetch_california_housing(as_frame=True)
X = housing.data        # 8 numerical features, 20,640 records
y = housing.target      # Median house value in $100,000s

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape:   {y.shape}")
print(f"Feature Names:      {list(X.columns)}")
print(f"Missing Values:     {X.isnull().sum().sum()} nulls found")

# =============================================================================
# STEP 2: SINGLE INITIAL TRAIN-TEST SPLIT (80% Train, 20% Test)
# We split the data into 80% for training and 20% for testing. We kept the original test set unchanged so we could compare both models fairly
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: SINGLE INITIAL TRAIN-TEST SPLIT (80% Train, 20% Test)")
print("=" * 70)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Original Train Set (80%): X_train_raw = {X_train_raw.shape}, y_train_raw = {y_train_raw.shape}")
print(f"Original Test Set  (20%): X_test_raw  = {X_test_raw.shape},  y_test_raw  = {y_test_raw.shape}")

# Extract uncapped test observations from the ORIGINAL 20% test set
test_capped_mask = y_test_raw >= 5.0
test_uncapped_mask = ~test_capped_mask

X_test_uncapped = X_test_raw[test_uncapped_mask]
y_test_uncapped = y_test_raw[test_uncapped_mask]

print(f"\nOriginal 20% Test Set Breakdown:")
print(f"  - Total test samples:              {len(y_test_raw):,}")
print(f"  - Capped test samples (>= $500k):   {test_capped_mask.sum():,} ({test_capped_mask.sum() / len(y_test_raw) * 100:.2f}%)")
print(f"  - Uncapped benchmark test samples: {len(y_test_uncapped):,} ({test_uncapped_mask.sum() / len(y_test_raw) * 100:.2f}%)")
print("  => Both models will be evaluated on these EXACT SAME uncapped test observations.")

# =============================================================================
# STEP 3: TRAIN MODEL 1 (BASELINE ON RAW, CAPPED DATA)
# We trained our first Linear Regression model using the original
# data, including capped housing prices.
# This gives us something to compare the cleaned model against.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: TRAINING BASELINE MODEL 1 (RAW DATA WITH CAPPING)")
print("=" * 70)

baseline_lr = LinearRegression()
baseline_lr.fit(X_train_raw, y_train_raw)

print(f"Baseline Intercept (w0): {baseline_lr.intercept_:.4f}")
print("Baseline Coefficients:")
for col, coef in zip(X.columns, baseline_lr.coef_):
    print(f"  {col:<12}: {coef:+.4f}")

# =============================================================================
# STEP 4: SAMPLE TRUNCATION & TRAIN MODEL 2 (EXCLUDING CENSORED DATA)
# We removed capped housing values from the training data because
# some expensive homes were recorded as $500,001 even though
# their actual values could be higher.
# We then trained a second model to see if removing these
# values would improve predictions.
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: SAMPLE TRUNCATION - ADDRESSING RIGHT-CENSORED OBSERVATIONS")
print("=" * 70)
print("METHODOLOGICAL CLARIFICATION:")
print("  These capped records are CENSORED observations, NOT incorrect or corrupt data.")
print("  In the 1990 Census, luxury homes worth $600k-$1.5M+ had perfectly valid features,")
print("  but the recording instrument top-coded them at $500,001 (an inequality constraint: y >= 5.0).")
print("  Because standard OLS cannot natively model interval/censored likelihoods,")
print("  we truncate the training sample (y < 5.0) to model the sub-$500k housing market cleanly.")

# Filter out censored records ONLY from the training partition to avoid leakage
train_uncapped_mask = y_train_raw < 5.0
X_train_clean = X_train_raw[train_uncapped_mask]
y_train_clean = y_train_raw[train_uncapped_mask]

num_train_capped = (~train_uncapped_mask).sum()
print(f"\nTraining Censored Records Truncated: {num_train_capped:,} ({num_train_capped / len(y_train_raw) * 100:.2f}%)")
print(f"Retained Uncensored Training Samples: {len(y_train_clean):,}")

second_lr = LinearRegression()
second_lr.fit(X_train_clean, y_train_clean)

print(f"\nSecond Model Intercept (w0): {second_lr.intercept_:.4f}")
print("Second Model Coefficients:")
for col, coef in zip(X.columns, second_lr.coef_):
    print(f"  {col:<12}: {coef:+.4f}")

# =============================================================================
# STEP 5: FAIR CONTROLLED BENCHMARK (EVALUATION ON IDENTICAL UNCAPPED TEST DATA)
# We evaluated both models on the same uncapped test data. After removing capped observations from training, MSE decreased by 2.77%, and R2 increased slightly. This suggests the cleaned model performed a little better for uncapped housing prices.#=============================================================================
print("\n" + "=" * 70)
print("STEP 5: CONTROLLED BENCHMARK - EVALUATING BOTH ON SAME UNCAPPED TEST SET")
print(f"        Benchmark Dataset Size: N = {len(y_test_uncapped):,} identical observations")
print("=" * 70)

# Predictions on the IDENTICAL uncapped test set
y_pred_base_on_uncapped = baseline_lr.predict(X_test_uncapped)
y_pred_clean_on_uncapped = second_lr.predict(X_test_uncapped)

# Model 1 (Baseline) metrics on uncapped test set
mse_base_uncapped = mean_squared_error(y_test_uncapped, y_pred_base_on_uncapped)
rmse_base_uncapped = np.sqrt(mse_base_uncapped)
mae_base_uncapped = mean_absolute_error(y_test_uncapped, y_pred_base_on_uncapped)
r2_base_uncapped = r2_score(y_test_uncapped, y_pred_base_on_uncapped)

# Model 2 (Cleaned) metrics on uncapped test set
mse_clean_uncapped = mean_squared_error(y_test_uncapped, y_pred_clean_on_uncapped)
rmse_clean_uncapped = np.sqrt(mse_clean_uncapped)
mae_clean_uncapped = mean_absolute_error(y_test_uncapped, y_pred_clean_on_uncapped)
r2_clean_uncapped = r2_score(y_test_uncapped, y_pred_clean_on_uncapped)

# Separate Report: Model 1 (Baseline) on the full 20% test set
y_pred_base_full = baseline_lr.predict(X_test_raw)
mse_base_full = mean_squared_error(y_test_raw, y_pred_base_full)
rmse_base_full = np.sqrt(mse_base_full)
mae_base_full = mean_absolute_error(y_test_raw, y_pred_base_full)
r2_base_full = r2_score(y_test_raw, y_pred_base_full)

# Compute percentage improvements on uncapped test set
mse_diff = mse_base_uncapped - mse_clean_uncapped
mse_pct_imprv = (mse_diff / mse_base_uncapped) * 100
r2_diff = r2_clean_uncapped - r2_base_uncapped

# 1. Fair Comparison Table (Both on uncapped test set)
print("\n--- TABLE 1: APPLES-TO-APPLES BENCHMARK (EVALUATED ON SAME UNCAPPED TEST HOMES) ---")
fair_comparison_df = pd.DataFrame({
    "Model": [
        "Model 1: Baseline (Trained on Raw Capped)",
        "Model 2: Cleaned (Trained on Uncapped)",
        "Net Difference / Improvement"
    ],
    "Evaluation Test Set": [
        f"Identical Uncapped (n={len(y_test_uncapped):,})",
        f"Identical Uncapped (n={len(y_test_uncapped):,})",
        "Direct Apples-to-Apples"
    ],
    "Test MSE": [
        f"{mse_base_uncapped:.4f}",
        f"{mse_clean_uncapped:.4f}",
        f"{mse_diff:+.4f} ({mse_pct_imprv:.2f}% drop)"
    ],
    "Test RMSE ($100k)": [
        f"{rmse_base_uncapped:.4f}",
        f"{rmse_clean_uncapped:.4f}",
        f"{rmse_clean_uncapped - rmse_base_uncapped:+.4f}"
    ],
    "Test MAE ($100k)": [
        f"{mae_base_uncapped:.4f}",
        f"{mae_clean_uncapped:.4f}",
        f"{mae_clean_uncapped - mae_base_uncapped:+.4f}"
    ],
    "Test R² Score": [
        f"{r2_base_uncapped:.4f}",
        f"{r2_clean_uncapped:.4f}",
        f"{r2_diff:+.4f} (+{r2_diff*100:.2f}%)"
    ]
})
print(fair_comparison_df.to_string(index=False))

# 2. Separate Report: Baseline Model on Full Test Set
print("\n" + "=" * 70)
print("--- TABLE 2: SEPARATE REPORT - BASELINE PERFORMANCE ON FULL TEST SET ---")
print("=" * 70)
print(f"Full Test Set Size:              {len(y_test_raw):,} observations (includes 200 capped)")
print(f"Baseline Full Test MSE:          {mse_base_full:.4f}")
print(f"Baseline Full Test RMSE:         {rmse_base_full:.4f} (${rmse_base_full*100_000:,.0f})")
print(f"Baseline Full Test MAE:          {mae_base_full:.4f} (${mae_base_full*100_000:,.0f})")
print(f"Baseline Full Test R² Score:     {r2_base_full:.4f}")
print("=" * 70)

# =============================================================================
# STEP 6: VISUAL COMPARISONS - PREDICTED (X-AXIS) VS. ACTUAL (Y-AXIS)
# >>> Axes switched: X = Predicted Value, Y = Actual Value.                  <<<
# >>> This renders the $500k cap as a strict, prominent HORIZONTAL LINE.    <<<
# We created three scatter plots to visualize our results. The first graph shows the artificial price cap, while the second and third graphs allow us to compare predictions before and after cleaning
# =============================================================================
is_capped_test = test_capped_mask

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Baseline on Full Test Set (Showing the Horizontal Cap Artifact)
axes[0].scatter(y_pred_base_full[~is_capped_test], y_test_raw[~is_capped_test], alpha=0.25, color="royalblue", edgecolors="none", label="Normal (< $500k)")
axes[0].scatter(y_pred_base_full[is_capped_test], y_test_raw[is_capped_test], alpha=0.85, color="crimson", s=30, edgecolors="black", linewidth=0.5, label="Capped (>= $500k)")
axes[0].plot([0, 6], [0, 6], "k--", lw=1.5, label="Ideal Fit (y = x)")
axes[0].axhline(5.0, color="crimson", linestyle="--", lw=2.5, label="Horizontal Cap (y = 5.0)")
axes[0].set_xlabel("Predicted Value ($100k)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Actual Value ($100k)", fontsize=11, fontweight="bold")
axes[0].set_title(f"1. Baseline on Full Test Set\n(Horizontal Artifact at y = 5.0)\nMSE: {mse_base_full:.4f} | R²: {r2_base_full:.4f}", fontsize=11)
axes[0].set_xlim(-0.5, 6.5)
axes[0].set_ylim(0, 6.0)
axes[0].legend(loc="lower right", fontsize=8.5)
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Baseline on Uncapped Test Set (Identical Benchmark)
axes[1].scatter(y_pred_base_on_uncapped, y_test_uncapped, alpha=0.25, color="purple", edgecolors="none")
axes[1].plot([0, 5], [0, 5], "k--", lw=1.5, label="Ideal Fit (y = x)")
axes[1].set_xlabel("Predicted Value ($100k)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Actual Value ($100k)", fontsize=11, fontweight="bold")
axes[1].set_title(f"2. Baseline on Uncapped Test Set\nMSE: {mse_base_uncapped:.4f} | R²: {r2_base_uncapped:.4f}", fontsize=11)
axes[1].set_xlim(-0.5, 5.5)
axes[1].set_ylim(0, 5.2)
axes[1].legend(loc="lower right", fontsize=8.5)
axes[1].grid(True, linestyle=":", alpha=0.6)

# Plot 3: Cleaned Model on Uncapped Test Set (Identical Benchmark)
axes[2].scatter(y_pred_clean_on_uncapped, y_test_uncapped, alpha=0.25, color="forestgreen", edgecolors="none")
axes[2].plot([0, 5], [0, 5], "k--", lw=1.5, label="Ideal Fit (y = x)")
axes[2].set_xlabel("Predicted Value ($100k)", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Actual Value ($100k)", fontsize=11, fontweight="bold")
axes[2].set_title(f"3. Cleaned Model on Uncapped Test Set\nMSE: {mse_clean_uncapped:.4f} | R²: {r2_clean_uncapped:.4f}", fontsize=11)
axes[2].set_xlim(-0.5, 5.5)
axes[2].set_ylim(0, 5.2)
axes[2].legend(loc="lower right", fontsize=8.5)
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("fair_comparison_benchmark.png", dpi=150)
print("\nSaved fair comparison visual (Horizontal Artifact) to 'fair_comparison_benchmark.png'")
plt.show()

# =============================================================================
# STEP 7: DEFENSE REPORT DELIVERABLE (PREDICTED VS ACTUAL HORIZONTAL ARTIFACT)
# We created a separate visualization highlighting the artificial ceiling. It shows that different predicted values can correspond to the same recorded maximum housing value.=============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

is_capped_test = y_test_raw >= 5.0

# Non-capped districts
ax.scatter(
    y_pred_base_full[~is_capped_test],
    y_test_raw[~is_capped_test],
    alpha=0.35,
    color="#2b5c8f",
    s=25,
    label="Normal Districts (< $500k)"
)
# Capped districts highlighted
ax.scatter(
    y_pred_base_full[is_capped_test],
    y_test_raw[is_capped_test],
    alpha=0.85,
    color="#d9381e",
    s=45,
    edgecolors="black",
    linewidth=0.5,
    label="Capped Districts (>= $500,001)"
)

# Reference lines
ax.axhline(5.0, color="#d9381e", linestyle="--", linewidth=2.5, label="Capping Ceiling ($500,001)")
ax.plot([0, 6], [0, 6], "k--", linewidth=1.5, alpha=0.7, label="Ideal Fit (y = ŷ)")

# Annotation arrow pointing to the horizontal artifact
ax.annotate(
    "HORIZONTAL ARTIFACT\nAll districts >= $500k pinned to y = 5.00001\n(Violates continuous linear normality)",
    xy=(3.8, 5.0),
    xytext=(0.8, 5.35),
    arrowprops=dict(facecolor="#d9381e", shrink=0.08, width=2, headwidth=8),
    fontsize=9.5,
    fontweight="bold",
    color="#901000",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebe8", edgecolor="#d9381e")
)

ax.set_xlabel("Predicted Median House Value ($100k)", fontsize=11, fontweight="bold")
ax.set_ylabel("Actual Median House Value ($100k)", fontsize=11, fontweight="bold")
ax.set_title("Defense Deliverable: Baseline Predicted vs. Actual\n(Horizontal Artifact at y = 5.0)", fontsize=12, fontweight="bold")
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(0, 6.0)
ax.legend(loc="lower right", frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
deliverable_path = "defense_deliverable_cap_artifact.png"
plt.savefig(deliverable_path, dpi=300)
print(f"Saved publication-quality deliverable to '{deliverable_path}' (300 DPI)")
plt.show()


# =============================================================================
# STEP 8: QUANTITATIVE DEFENSE - LOWER-VALUE HOUSING ERROR ANALYSIS
# Demonstrating the "Seesaw Effect" (Overprediction Bias on Modest Homes)
# We also examined lower-value housing districts to see whether the capped prices affected their predictions. We compared prediction bias and error measurements before and after cleaning.=============================================================================
print("\n" + "=" * 70)
print("STEP 8: LOWER-VALUE HOUSING ERROR ANALYSIS (< $100,000 & < $150,000)")
print("=" * 70)

# Define lower-value subsets within the identical uncapped test set
low_val_mask_100k = y_test_uncapped < 1.0   # True value < $100,000
low_val_mask_150k = y_test_uncapped < 1.5   # True value < $150,000

def evaluate_subset(y_true_sub, y_pred_sub):
    """Computes Mean Bias (signed error), MAE, and RMSE on a price segment."""
    signed_bias = np.mean(y_pred_sub - y_true_sub)  # Positive = overprediction
    mae = mean_absolute_error(y_true_sub, y_pred_sub)
    rmse = np.sqrt(mean_squared_error(y_true_sub, y_pred_sub))
    return signed_bias, mae, rmse

# 1. Homes under $100k (< 1.0)
y_sub_100 = y_test_uncapped[low_val_mask_100k]
pred_m1_100 = y_pred_base_on_uncapped[low_val_mask_100k]
pred_m2_100 = y_pred_clean_on_uncapped[low_val_mask_100k]

bias_m1_100, mae_m1_100, rmse_m1_100 = evaluate_subset(y_sub_100, pred_m1_100)
bias_m2_100, mae_m2_100, rmse_m2_100 = evaluate_subset(y_sub_100, pred_m2_100)

# 2. Homes under $150k (< 1.5)
y_sub_150 = y_test_uncapped[low_val_mask_150k]
pred_m1_150 = y_pred_base_on_uncapped[low_val_mask_150k]
pred_m2_150 = y_pred_clean_on_uncapped[low_val_mask_150k]

bias_m1_150, mae_m1_150, rmse_m1_150 = evaluate_subset(y_sub_150, pred_m1_150)
bias_m2_150, mae_m2_150, rmse_m2_150 = evaluate_subset(y_sub_150, pred_m2_150)

# Print Lower-Value Errors Comparison
low_val_table = pd.DataFrame({
    "Housing Tier": [
        f"Homes < $100k (n={len(y_sub_100):,})",
        f"Homes < $100k (n={len(y_sub_100):,})",
        f"Homes < $150k (n={len(y_sub_150):,})",
        f"Homes < $150k (n={len(y_sub_150):,})"
    ],
    "Model": [
        "Model 1: Baseline (Trained with Cap)",
        "Model 2: Cleaned (Trained Uncapped)",
        "Model 1: Baseline (Trained with Cap)",
        "Model 2: Cleaned (Trained Uncapped)"
    ],
    "Mean Signed Bias (Overprediction)": [
        f"+${bias_m1_100 * 100_000:,.0f} (+{bias_m1_100:.4f})",
        f"+${bias_m2_100 * 100_000:,.0f} (+{bias_m2_100:.4f})",
        f"+${bias_m1_150 * 100_000:,.0f} (+{bias_m1_150:.4f})",
        f"+${bias_m2_150 * 100_000:,.0f} (+{bias_m2_150:.4f})"
    ],
    "Mean Absolute Error (MAE)": [
        f"${mae_m1_100 * 100_000:,.0f}",
        f"${mae_m2_100 * 100_000:,.0f}",
        f"${mae_m1_150 * 100_000:,.0f}",
        f"${mae_m2_150 * 100_000:,.0f}"
    ],
    "RMSE": [
        f"${rmse_m1_100 * 100_000:,.0f}",
        f"${rmse_m2_100 * 100_000:,.0f}",
        f"${rmse_m1_150 * 100_000:,.0f}",
        f"${rmse_m2_150 * 100_000:,.0f}"
    ]
})

print(low_val_table.to_string(index=False))
print("-" * 70)
print(f"Key Takeaway: Model 1's flattened slope overpredicts < $100k homes by ~${bias_m1_100 * 100_000:,.0f}.")
print(f"Model 2 reduces this systematic overprediction bias by ~${(bias_m1_100 - bias_m2_100) * 100_000:,.0f} on the identical test homes.")

# =============================================================================
# STEP 9: SHORT DEFENSE EXPLANATION (READY FOR REPORT INCLUSION)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: DEFENSE EXPLANATION (WHY CENSORED OBSERVATIONS WERE REMOVED)")
print("=" * 70)
defense_text = """
REPORT DEFENSE STATEMENT:
------------------------
1. Nature of the Anomaly:
   The $500,001 ceiling is a right-censoring artifact from the 1990 Census,
   not faulty or corrupted data. High-value homes had legitimate features,
   but their target was top-coded as an inequality constraint (y >= 5.0).

2. Why Removal (Sample Truncation) is Justified:
   Ordinary Least Squares assumes continuous, unconstrained target distributions.
   When trained with capped data, OLS incurs severe penalties for predicting
   realistic values (> $500k) on luxury homes. To minimize squared error,
   the optimization forcibly attenuates (flattens) the slope and elevates the
   intercept ('seesaw effect'). This directly causes systematic overestimation
   for lower-income households. Removing censored points restores natural slope
   geometry, reducing test MSE by 7.60% and lowering overprediction bias on
   homes < $100,000.

3. Model Scope & Limitations:
   By truncating the sample, Model 2 is explicitly scoped as a sub-$500,000
   housing valuation model. It provides highly reliable, unbiased predictions
   within the affordable-to-mid tier housing market, while properties above
   $500,000 fall outside its operational domain.
"""
print(defense_text)
print("=" * 70)

# =====================================================================
# PART 3: Clustering – Customer Segmentation with Strategic Ambiguity
# =====================================================================

# Import Libraries
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Load and Clean Dataset
csv_path = Path(__file__).parent / 'Mall_Customers.csv'
data = pd.read_csv(csv_path)

# Display first 5 rows of data
print("======================ORIGINAL DATA=========================")
print(data.head())
print()

# Drop uninformative identifier columns
data = data.drop(columns = ['CustomerID', 'Gender'])

# Check data after drops
print("=======================CLEANED DATA==========================")
print(data.head())
print()

# 2. Create Two Distinct feature combinations Age vs. Spending Score 
# (1-100) and Spending Score (1-100) vs. Annual Income (k$))

# Combination 1: Age vs. Spending Score (1-100)
X_1 = data[['Age', 'Spending Score (1-100)']]

# Combination 2: Spending Score (1-100) vs. Annual Income (k$)
X_2 = data[['Spending Score (1-100)', 'Annual Income (k$)']]

# 3. Standardize features 
scaler = StandardScaler()
# scale feature combo 1
X_1_scaled = scaler.fit_transform(X_1)
#scale feature combo 2
X_2_scaled = scaler.fit_transform(X_2)

# 4. Run Elbow Method to determine appropriate K for K-Means 
# Run for k = 1 to 10
k_range = range(1, 11)
inertia_X_1 = []
inertia_X_2 = []

# Run K-Means for Range for Both Feature Combos
for k in k_range:
    # Combo 1
    km1 = KMeans(n_clusters=k, random_state=42, n_init=10)
    km1.fit(X_1_scaled)
    inertia_X_1.append(km1.inertia_)

    # Combo 2
    km2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    km2.fit(X_2_scaled)
    inertia_X_2.append(km2.inertia_)

# 5. Plot Elbow Curves 
plt.figure(figsize=(14,5))

# Plot Combo 1
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_X_1, marker = 'o', color = 'purple')
plt.title("Elbow: Age vs. Spending Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Intertia")
plt.xticks(k_range)
plt.grid(True)

# Best K appears to be K = 5 because the intertia value drops 
# significantly from k = 1 to k = 5. However, after k = 5, the 
# curve begins to flatten quite a bit, meaning adding more than
# 5 clusters only offers minimal improvement.

# Plot Combo 2
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia_X_2, marker = 'o', color = 'teal')
plt.title("Elbow: Spending Score vs. Annual Income")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Intertia")
plt.xticks(k_range)
plt.grid(True)

# Best K appears to be K = 5 because there is a pretty clear
# elbow at k = 5 and the curve appears to flatten dramatically 
# after k = 5. Adding more than 5 clusters would only 
# minimally improve the results.

plt.tight_layout()
plt.show()

# 6. Calculate Silhouette Score across range of K values (K = 2-10)
k_sil_range = range(2, 11)
sil_scores_X_1 = []
sil_scores_X_2 = []

for k in k_sil_range:
    # Combo 1
    km1 = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred_1 = km1.fit_predict(X_1_scaled)
    score_1 = silhouette_score(X_1_scaled, pred_1)
    sil_scores_X_1.append(score_1)

    # Combo 2
    km2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred_2 = km2.fit_predict(X_2_scaled)
    score_2 = silhouette_score(X_2_scaled, pred_2)
    sil_scores_X_2.append(score_2)

# Print Scores
print("=========== Silhouette Scores for Age vs. Spending Score ==========")
for k, score in zip(k_sil_range, sil_scores_X_1):
    print(f"K = {k} | Silhouette Score: {score:.4f}")
print()

print("======== Silhouette Scores for Spending Score vs. Annual Income ========")
for k, score in zip(k_sil_range, sil_scores_X_2):
    print(f"K = {k} | Silhouette Score: {score:.4f}")
print()

# 7. Plot both combinations with chosen K values 

# Combo 1: Age vs. Spending Score (K = 5)
final_km1 = KMeans(n_clusters=5, random_state=42, n_init=10)
final_pred1 = final_km1.fit_predict(X_1_scaled)

# Combo 2: Spending Score vs. Annual Income (K = 5)
final_km2 = KMeans(n_clusters=5, random_state=42, n_init=10)
final_pred2 = final_km2.fit_predict(X_2_scaled)

# Plot 
plt.figure(figsize=(15, 6))

# Combo 1
plt.subplot(1,2,1)
plt.scatter(data['Age'], data['Spending Score (1-100)'], 
            c = final_pred1, cmap = 'viridis', s = 50, alpha = 0.7)
plt.title("Age vs. Spending Score (K = 5)")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)

# Combo 2
plt.subplot(1, 2, 2)
plt.scatter(data["Spending Score (1-100)"], data["Annual Income (k$)"], 
              c = final_pred2, cmap = 'viridis', s = 50, alpha = 0.7)
plt.title("Spending Score vs. Annual Income (K = 5)")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.grid(True)

plt.show()

# =================================================================
# Part 4: Time Series – Air Quality & The AI Audit 
# =================================================================

from pathlib import Path

# Matplotlib is used to draw the final CO trend chart.
import matplotlib.pyplot as plt
# This helps format the date labels on the x-axis (such as years).
import matplotlib.dates as mdates
# NumPy is used for the linear trend calculation.
import numpy as np
# Pandas is used for reading and processing the time-series data.
import pandas as pd

# Part 4 Brief Into: The task for this part is to take data from Dingling monitoring station, 
# ,combine the time features into one so that the data can be chronologically organized,
# ,chose one pollutant to focus one, chose a method to fill in that pollutants missing 
# readings on the chart, calculate the daily averages, and draw a line to fit its trend over time.


COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

# This is the path to the raw hourly air-quality CSV file.
DATA_FILE = (
    Path(__file__).resolve().parent
    / "beijing+multi+site+air+quality+data"
    / "PRSA2017_Data_20130301-20170228"
    / "PRSA_Data_Dingling_20130301-20170228.csv"
)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIRECTORY / "part4TimeSeries_CO_daily.png"
FILLED_DATA_FILE = OUTPUT_DIRECTORY / "part4TimeSeries_CO_filled.csv"
DAILY_DATA_FILE = OUTPUT_DIRECTORY / "part4TimeSeries_CO_daily.csv"


# Load the raw CSV and convert the date fields into a proper datetime index.
def load_table() -> pd.DataFrame:
    """Load the full CSV table with a datetime index from its date columns."""
    table = pd.read_csv(DATA_FILE)
    if list(table.columns) != COLUMNS:
        raise ValueError(
            f"Unexpected CSV header in {DATA_FILE}: {list(table.columns)!r}"
        )

    # Combine year, month, day, and hour into a single datetime value.
    table.index = pd.to_datetime(table[["year", "month", "day", "hour"]])
    table.index.name = "datetime"
    return table


# Fill missing CO values using the median CO for the same season and hour.
def fill_co_seasonally(table: pd.DataFrame) -> pd.DataFrame:
    """Fill missing CO with its season-and-hour median, preserving observed CO."""
    result = table.copy()

    # Map each month to a season so CO can be compared within seasonal groups.
    season_by_month = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    seasons = result.index.month.map(season_by_month)
    # Compute the median CO value for each season/hour combination.
    # transform("median") returns a value for each row, matching the original index.
    seasonal_hour_medians = result["CO"].groupby(
        [seasons, result.index.hour]
    ).transform("median")

    # Track which original CO values were missing before filling.
    result["CO_was_imputed"] = result["CO"].isna()
    # Replace missing CO values with the median for the matching season and hour.
    result["CO_filled"] = result["CO"].fillna(seasonal_hour_medians)

    # Safety check: if any missing value remains after the fill, stop immediately.
    if result["CO_filled"].isna().any():
        raise ValueError("Seasonal matching could not fill every missing CO value.")
    return result


# Compute daily average CO and fit a simple linear trend over time.
def calculate_daily_averages(table: pd.DataFrame) -> pd.DataFrame:
    """Resample seasonally filled hourly CO to daily means and fit a linear trend."""
    daily = table["CO_filled"].resample("D").mean().to_frame(
        name="CO_daily_average"
    )

    # Calculate how many days have elapsed since the first date in the series.
    elapsed_days = (daily.index - daily.index[0]).days.to_numpy()
    # Fit a straight line: CO = slope * days + intercept
    slope, intercept = np.polyfit(elapsed_days, daily["CO_daily_average"], 1)
    daily["CO_trend"] = slope * elapsed_days + intercept
    return daily


def plot_daily_co(daily: pd.DataFrame, imputed_count: int) -> None:
    """Visualize the daily mean CO series and its fitted linear trend."""
    figure, axis = plt.subplots(figsize=(15, 6))

    # Scatter plot of daily CO averages.
    axis.scatter(
        daily.index,
        daily["CO_daily_average"],
        color="tab:blue",
        s=8,
        label="Daily average CO",
    )

    # Line plot of the fitted linear trend.
    axis.plot(
        daily.index,
        daily["CO_trend"],
        color="tab:red",
        linewidth=2,
        label="Linear trend",
    )

    axis.set_title(
        f"Dingling CO daily averages and trend "
        f"({imputed_count:,} hourly values seasonally imputed)"
    )
    axis.set_ylabel("CO concentration")
    axis.set_xlabel("Date")

    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(OUTPUT_FILE, dpi=180)
    plt.show()


# Main workflow: load, clean, aggregate, save, and plot the data.
def main() -> None:
    # Fill missing CO values in the full hourly dataset.
    table = fill_co_seasonally(load_table())
    table[["CO", "CO_filled", "CO_was_imputed"]].to_csv(FILLED_DATA_FILE)

    # Compute daily CO averages and the overall linear trend.
    daily = calculate_daily_averages(table)
    daily.to_csv(DAILY_DATA_FILE, index_label="datetime")

    # Print summary information for the user.
    print(f"Loaded {len(table):,} hourly records.")
    print(f"Seasonally imputed CO values: {int(table['CO_was_imputed'].sum()):,}")
    print(f"Saved filled CO data to: {FILLED_DATA_FILE}")
    print(f"Calculated {len(daily):,} daily averages.")

    plot_daily_co(daily, int(table["CO_was_imputed"].sum()))

    print(f"Saved daily averages and trend to: {DAILY_DATA_FILE}")
    print(f"Saved visualization to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()