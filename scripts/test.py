import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap
from scipy.stats import entropy


def cli():
    """
    Command Line Interface

    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", type=str, required=True,
                        help="Input data file 1 in CSV format.")
    parser.add_argument("-i2", "--input2", type=str, required=True,
                        help="Input data file 2 in TSV format.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory to store results.")
    return parser.parse_args()


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


def calculate_qed(smiles_list):
    qed_scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            qed_score = QED.qed(mol)
            qed_scores.append(qed_score)
        else:
            qed_scores.append(None)
    return qed_scores


def calculate_molecular_properties(mols):
    molecular_weights = [Descriptors.MolWt(mol) for mol in mols]
    logp_values = [Descriptors.MolLogP(mol) for mol in mols]
    return molecular_weights, logp_values


def main(input_file1, input_file2, output_dir):
    # Load data from CSV
    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2, delimiter='\t')

    # Extract SMILES
    smiles_list1 = data1['smiles'].tolist()
    smiles_list2 = data2['SMILES'].tolist()

    # Original Molecules
    original_molecules1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1]
    antibacterial1 = data1[data1['antibacterial'] == 1]['smiles'].tolist()
    non_antibacterial1 = data1[data1['antibacterial'] == 0]['smiles'].tolist()

    # New Molecules
    new_molecules2 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list2]

    # Calculate molecular properties
    weights1, logp1 = calculate_molecular_properties(antibacterial1)
    weights2, logp2 = calculate_molecular_properties(new_molecules2)

    # Combine the molecular properties for dimensionality reduction
    combined_properties = np.array(weights1 + weights2 + logp1 + logp2)

    # Define dimensionality reduction processes
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    for reduction_method in reduction_methods:
        # Perform dimensionality reduction
        reduced_data = perform_dimensionality_reduction(combined_properties, reduction_method)

        # Split the reduced data
        reduced_data1 = reduced_data[:len(original_molecules1)]
        reduced_data2 = reduced_data[len(original_molecules1):len(original_molecules1) + len(new_molecules2)]

        # Plot original antibacterial molecules, original non-antibacterial molecules, and new molecules
        plt.figure(figsize=(12, 6))

        plt.scatter(reduced_data1[:len(antibacterial1), 0], reduced_data1[:len(antibacterial1), 1],
                    label='Original Antibacterial', color='blue', alpha=0.6)

        plt.scatter(reduced_data1[len(non_antibacterial1):, 0], reduced_data1[len(non_antibacterial1):, 1],
                    label='Original Non-Antibacterial', color='red', alpha=0.6)

        plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1],
                    label='New Molecules', color='black', alpha=0.6)

        plt.xlabel(f'{reduction_method} Component 1')
        plt.ylabel(f'{reduction_method} Component 2')
        plt.title(f'Original vs New Molecules with {reduction_method}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, f"original_vs_new_{reduction_method}.png"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    args = cli()
    main(args.input1, args.input2, args.output)
