import argparse
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import matplotlib.colors as mcolors
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def cli() -> argparse.Namespace:
    """
    Command Line Interface

    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input data file in CSV format.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory to store results.")
    return parser.parse_args()

def perform_clustering(data, method):
    if method == "KMeans":
        cluster_alg = KMeans(n_clusters=10, random_state=42)
    elif method == "Agglomerative":
        cluster_alg = AgglomerativeClustering(n_clusters=10)
    else:
        raise ValueError("Invalid clustering method.")

    labels = cluster_alg.fit_predict(data)
    return labels


def perform_dimensionality_reduction(data, method):
    if method == "PCA":
        dim_red = PCA(n_components=2)
    elif method == "MDS":
        dim_red = MDS(n_components=2, random_state=42)
    elif method == "t-SNE":
        dim_red = TSNE(n_components=2, random_state=42)
    elif method == "UMAP":
        dim_red = umap.UMAP(n_components=2)
    else:
        raise ValueError("Invalid dimensionality reduction method.")

    reduced_data = dim_red.fit_transform(data)
    return reduced_data


def save_cluster_info_to_csv(data, labels, output_dir, method, reduction):
    data['cluster'] = labels
    output_filename = os.path.join(output_dir, f"cluster_info_{method}_{reduction}.csv")
    data.to_csv(output_filename, index=False)


def main(input_file, output_dir):
    # Load data from CSV
    data = pd.read_csv(input_file)

    # Extract data
    smiles_list = data['smiles'].tolist()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]

    # Convert fingerprints to array
    fp_array = np.zeros((len(fingerprints), 1024))
    for i, fp in enumerate(fingerprints):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[i] = arr

    # Define algorithms and dimensionality reduction processes
    clustering_methods = ["KMeans", "Agglomerative"]
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    # Generate plot
    num_clusters = 10
    cluster_colors = [mcolors.to_rgba(f"C{i}") for i in range(num_clusters)]

    for cluster_method in clustering_methods:
        for reduction_method in reduction_methods:
            # Perform dimensionality reduction
            reduced_data = perform_dimensionality_reduction(fp_array, reduction_method)

            # Perform clustering algorithm
            labels = perform_clustering(reduced_data, cluster_method)

            # Plot the results
            plt.figure(figsize=(12, 6))
            unique_labels = np.unique(labels)
            for i, color in zip(unique_labels, cluster_colors):
                plt.scatter(reduced_data[labels == i, 0], reduced_data[labels == i, 1], label=f'Cluster {i + 1}',
                            color=color)
            plt.xlabel(f'{reduction_method} Component 1')
            plt.ylabel(f'{reduction_method} Component 2')
            plt.title(f'Molecular Clustering with {cluster_method} and {reduction_method}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
            plt.savefig(os.path.join(output_dir, f"cluster_plots_{cluster_method}_{reduction_method}.png"))
            plt.close()

            # Save cluster information to CSV
            save_cluster_info_to_csv(data, labels, output_dir, cluster_method, reduction_method)


if __name__ == "__main__":
    args = cli()
    main(args.input, args.output)
