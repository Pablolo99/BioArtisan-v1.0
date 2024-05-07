from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

# Load data from CSV
data = pd.read_csv("C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/crossref_data.csv")

# Extract SMILES strings from the DataFrame
smiles_list = data['smiles'].tolist()

# Convert SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Generate Morgan fingerprints
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]

# Convert fingerprints to numpy array
fp_array = []
for fp in fingerprints:
    arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    fp_array.append(arr)

# Dimensionality reduction with UMAP
reducer = UMAP(n_components=2)
reduced_data = reducer.fit_transform(fp_array)

"""
# Evaluate the optimal number of clusters using Davies–Bouldin index and Calinski-Harabasz index
db_scores = []
ch_scores = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(reduced_data)
    db_score = davies_bouldin_score(reduced_data, labels)
    ch_score = calinski_harabasz_score(reduced_data, labels)
    db_scores.append(db_score)
    ch_scores.append(ch_score)

# Plot Davies–Bouldin index and Calinski-Harabasz index to determine optimal number of clusters
plt.plot(range(2, 20), db_scores, label='Davies–Bouldin Index', marker='o')
plt.plot(range(2, 20), ch_scores, label='Calinski-Harabasz Index', marker='x')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Cluster Evaluation')
plt.legend()
plt.show()

# Perform k-means clustering with the optimal number of clusters
num_clusters = np.argmax(ch_scores) + 2
"""
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(reduced_data)

# Add cluster labels to the original DataFrame
data['cluster'] = labels + 1

# Save the DataFrame with cluster labels to a new file
data.to_csv("clustered_data.csv", index=False)

# Define a list of distinct colors
num_unique_clusters = len(np.unique(labels))
cluster_colors = np.random.rand(num_unique_clusters, 3)

# Plot the results
unique_labels = np.unique(labels)
for i, color in zip(unique_labels, cluster_colors):
    plt.scatter(reduced_data[labels == i, 0], reduced_data[labels == i, 1], label=f'Cluster {i+1}', color=color)

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('Molecular Clustering with UMAP')
plt.legend()
plt.show()
