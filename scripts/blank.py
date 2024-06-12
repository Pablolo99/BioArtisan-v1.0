# Required libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# Paths to the files
file1_path = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/antibacterial_PK.csv'
file2_path = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/output_PKtest_200'

# Step 1: Read and parse the data
data1 = pd.read_csv(file1_path)
data1 = data1[data1['antibacterial'] == 1]
data2 = pd.read_csv(file2_path, delimiter='\t')

# Step 2: Calculate molecular descriptors (fingerprints)
def calculate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    return fingerprints

# Calculate fingerprints for both datasets
fingerprints1 = calculate_fingerprints(data1['smiles'])
fingerprints2 = calculate_fingerprints(data2['SMILES'])

# Convert fingerprints to numpy arrays
import numpy as np

fp_array1 = np.array([np.array(fingerprint) for fingerprint in fingerprints1])
fp_array2 = np.array([np.array(fingerprint) for fingerprint in fingerprints2])

# Combine the fingerprints for dimensionality reduction
combined_fp_array = np.vstack([fp_array1, fp_array2])

# Step 3: Apply UMAP for dimensionality reduction
reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard')
embedding_umap = reducer_umap.fit_transform(combined_fp_array)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(combined_fp_array)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
embedding_tsne = tsne.fit_transform(combined_fp_array)

# Split the embeddings
embedding1_umap = embedding_umap[:len(fp_array1)]
embedding2_umap = embedding_umap[len(fp_array1):]

embedding1_pca = embedding_pca[:len(fp_array1)]
embedding2_pca = embedding_pca[len(fp_array1):]

embedding1_tsne = embedding_tsne[:len(fp_array1)]
embedding2_tsne = embedding_tsne[len(fp_array1):]

# Step 4: Calculate QED for the molecules in the second file
qed_values = [QED.qed(Chem.MolFromSmiles(smiles)) for smiles in data2['SMILES']]

# Step 5: Visualize the results
fig, axes = plt.subplots(3, 1, figsize=(10, 24))

# UMAP plot
axes[0].scatter(embedding1_umap[:, 0], embedding1_umap[:, 1], label='Dataset original', alpha=0.7)
axes[0].scatter(embedding2_umap[:, 0], embedding2_umap[:, 1], label='Dataset new', alpha=0.7)
axes[0].legend()
axes[0].set_title('UMAP projection of the chemical space')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')

# PCA plot
axes[1].scatter(embedding1_pca[:, 0], embedding1_pca[:, 1], label='Dataset original', alpha=0.7)
axes[1].scatter(embedding2_pca[:, 0], embedding2_pca[:, 1], label='Dataset new', alpha=0.7)
axes[1].legend()
axes[1].set_title('PCA projection of the chemical space')
axes[1].set_xlabel('PCA1')
axes[1].set_ylabel('PCA2')

# t-SNE plot
axes[2].scatter(embedding1_tsne[:, 0], embedding1_tsne[:, 1], label='Dataset original', alpha=0.7)
axes[2].scatter(embedding2_tsne[:, 0], embedding2_tsne[:, 1], label='Dataset new', alpha=0.7)
axes[2].legend()
axes[2].set_title('t-SNE projection of the chemical space')
axes[2].set_xlabel('t-SNE1')
axes[2].set_ylabel('t-SNE2')

plt.show()

# Output QED values for the second dataset
data2['QED'] = qed_values
print(data2[['ID', 'SMILES', 'QED']])