import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# EDA: Print first few rows and summary
print("First 5 rows of data:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.savefig('gender_distribution.png')
plt.close()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.close()

# Annual Income distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
plt.title('Annual Income (k$) Distribution')
plt.savefig('income_distribution.png')
plt.close()

# Spending Score distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
plt.title('Spending Score Distribution')
plt.savefig('spending_score_distribution.png')
plt.close()

# Encode Gender
df['Gender_Code'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Code']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_method.png')
plt.close()

# Choose optimal k (e.g., k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters (using Annual Income and Spending Score only)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    data=df, 
    palette='tab10', 
    s=60
)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, features.index('Annual Income (k$)')],
    centers[:, features.index('Spending Score (1-100)')],
    c='black', s=200, alpha=0.6, marker='X', label='Centroids'
)
plt.legend()
plt.savefig('customer_clusters.png')
plt.close()

# Cluster summary statistics (numeric columns only)
print("\nCluster Summary (means):")
print(df.groupby('Cluster').mean(numeric_only=True))

# Gender distribution by cluster
print("\nGender distribution by cluster:")
print(df.groupby(['Cluster', 'Gender']).size().unstack())

<<<<<<< HEAD
# Not saving the clustered file as per your request
=======
# Not saving the clustered file as per your request
>>>>>>> e9c1cf5 (ADD RESULTS)
