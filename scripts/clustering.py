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


def save_cluster_info_to_csv(data, labels, output_dir, method):
    data['cluster'] = labels + 1  # Adding 1 to the labels to start clusters from 1
    output_filename = os.path.join(output_dir, f"cluster_info_{method}.csv")
    data.to_csv(output_filename, index=False)
    return output_filename


def calculate_antibacterial_proportion(cluster_info_file, output_dir, method):
    data = pd.read_csv(cluster_info_file)
    num_clusters = data['cluster'].max()
    proportion_filename = os.path.join(output_dir, f"antibacterial_proportion_{method}.txt")

    with open(proportion_filename, 'w') as f:
        for i in range(1, num_clusters + 1):
            cluster_data = data[data['cluster'] == i]
            total_molecules = len(cluster_data)
            antibacterial_molecules = cluster_data['antibacterial'].notnull().sum()
            percentage = (antibacterial_molecules / total_molecules) * 100
            f.write(f"cluster_{i}_total_molecules: {total_molecules}\n")
            f.write(f"cluster_{i}_antibacterial_molecules: {antibacterial_molecules}\n")
            f.write(f"proportion_antibacteria_cluster_{i}: {percentage:.2f}\n")


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

    # Generate plots
    num_clusters = 10
    cluster_colors = [mcolors.to_rgba(f"C{i}") for i in range(num_clusters)]

    for cluster_method in clustering_methods:
        # Perform clustering
        labels = perform_clustering(fp_array, cluster_method)

        # Save cluster information to CSV
        cluster_info_file = save_cluster_info_to_csv(data.copy(), labels, output_dir, cluster_method)

        # Calculate and save antibacterial proportions to a separate file
        calculate_antibacterial_proportion(cluster_info_file, output_dir, cluster_method)

        for reduction_method in reduction_methods:
            # Perform dimensionality reduction
            reduced_data = perform_dimensionality_reduction(fp_array, reduction_method)

            # Plot the clusters without distinguishing antibacterial molecules
            plt.figure(figsize=(12, 6))
            unique_labels = np.unique(labels)
            for i, color in zip(unique_labels, cluster_colors):
                mask = (labels == i)
                plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], label=f'Cluster {i + 1}', color=color, alpha=0.6)

            plt.xlabel(f'{reduction_method} Component 1')
            plt.ylabel(f'{reduction_method} Component 2')
            plt.title(f'Molecular Clustering with {cluster_method} and {reduction_method}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
            plt.savefig(os.path.join(output_dir, f"cluster_plots_{cluster_method}_{reduction_method}.png"),
                        bbox_inches='tight')
            plt.close()

            # Use the same reduced data to plot antibacterial vs. non-antibacterial molecules
            plt.figure(figsize=(12, 6))
            antibacterial_mask = (data['antibacterial'].notnull())
            non_antibacterial_mask = (data['antibacterial'].isnull())

            plt.scatter(reduced_data[non_antibacterial_mask, 0], reduced_data[non_antibacterial_mask, 1],
                        label=f'Non-Antibacterial ({np.sum(non_antibacterial_mask)})', color='blue', alpha=0.6)

            plt.scatter(reduced_data[antibacterial_mask, 0], reduced_data[antibacterial_mask, 1],
                        label=f'Antibacterial ({np.sum(antibacterial_mask)})', color='red', alpha=0.6)

            plt.xlabel(f'{reduction_method} Component 1')
            plt.ylabel(f'{reduction_method} Component 2')
            plt.title(f'Antibacterial vs Non-Antibacterial Molecules with {reduction_method}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
            plt.savefig(os.path.join(output_dir, f"antibacterial_vs_nonantibacterial_{reduction_method}_{cluster_method}.png"),
                        bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    args = cli()
    main(args.input, args.output)