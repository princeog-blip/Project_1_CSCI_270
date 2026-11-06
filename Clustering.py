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


