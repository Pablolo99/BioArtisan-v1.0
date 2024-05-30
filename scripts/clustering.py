import argparse
import os
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import matplotlib.colors as mcolors
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from k_means_constrained import KMeansConstrained

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
    parser.add_argument("-n", "--n_clusters", type=int, required=True,
                        help="Number of clusters.")
    parser.add_argument("--size_min", type=int, default=20,
                        help="Minimum size of each cluster.")
    return parser.parse_args()


def perform_clustering(data, n_clusters, size_min):
    cluster_alg = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        random_state=42
    )
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
            antibacterial_molecules = (cluster_data['antibacterial'] == 1).sum()
            percentage = (antibacterial_molecules / total_molecules) * 100
            f.write(f"cluster_{i}_total_molecules: {total_molecules}\n")
            f.write(f"cluster_{i}_antibacterial_molecules: {antibacterial_molecules}\n")
            f.write(f"proportion_antibacteria_cluster_{i}: {percentage:.2f}\n")


def main(input_file, output_dir, n_clusters, size_min):
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

    # Define dimensionality reduction processes
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    # Perform clustering
    labels = perform_clustering(fp_array, n_clusters, size_min)

    # Save cluster information to CSV
    cluster_info_file = save_cluster_info_to_csv(data.copy(), labels, output_dir, "ConstrainedKMeans")

    # Calculate and save antibacterial proportions to a separate file
    calculate_antibacterial_proportion(cluster_info_file, output_dir, "ConstrainedKMeans")

    for reduction_method in reduction_methods:
        # Perform dimensionality reduction
        reduced_data = perform_dimensionality_reduction(fp_array, reduction_method)

        # Plot the clusters without distinguishing antibacterial molecules
        plt.figure(figsize=(12, 6))
        unique_labels = np.unique(labels)
        for i, color in zip(unique_labels, mcolors.TABLEAU_COLORS.values()):
            mask = (labels == i)
            plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], label=f'Cluster {i + 1}', color=color, alpha=0.6)

        plt.xlabel(f'{reduction_method} Component 1')
        plt.ylabel(f'{reduction_method} Component 2')
        plt.title(f'Molecular Clustering with ConstrainedKMeans and {reduction_method}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, f"cluster_plots_ConstrainedKMeans_{reduction_method}.png"),
                    bbox_inches='tight')
        plt.close()

        # Use the same reduced data to plot antibacterial vs. non-antibacterial molecules
        plt.figure(figsize=(12, 6))
        antibacterial_mask = (data['antibacterial'] == 1)
        non_antibacterial_mask = (data['antibacterial'] == 0)

        plt.scatter(reduced_data[non_antibacterial_mask, 0], reduced_data[non_antibacterial_mask, 1],
                    label=f'Non-Antibacterial ({np.sum(non_antibacterial_mask)})', color='blue', alpha=0.6)

        plt.scatter(reduced_data[antibacterial_mask, 0], reduced_data[antibacterial_mask, 1],
                    label=f'Antibacterial ({np.sum(antibacterial_mask)})', color='red', alpha=0.6)

        plt.xlabel(f'{reduction_method} Component 1')
        plt.ylabel(f'{reduction_method} Component 2')
        plt.title(f'Antibacterial vs Non-Antibacterial Molecules with {reduction_method}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, f"antibacterial_vs_nonantibacterial_{reduction_method}_ConstrainedKMeans.png"),
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    args = cli()
    main(args.input, args.output, args.n_clusters, args.size_min)
