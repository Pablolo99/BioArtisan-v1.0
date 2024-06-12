import argparse
import os
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED

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

def main(input_file1, input_file2, output_dir):
    # Load data from CSV
    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2, delimiter='\t')

    # Separate data1 into antibacterial and non-antibacterial
    data1_antibacterial = data1[data1['antibacterial'] == 1]
    data1_non_antibacterial = data1[data1['antibacterial'] == 0]

    # Extract data
    smiles_list1_antibacterial = data1_antibacterial['smiles'].tolist()
    smiles_list1_non_antibacterial = data1_non_antibacterial['smiles'].tolist()
    smiles_list2 = data2['SMILES'].tolist()

    molecules1_antibacterial = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1_antibacterial]
    molecules1_non_antibacterial = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1_non_antibacterial]
    molecules2 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list2]

    fingerprints1_antibacterial = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules1_antibacterial]
    fingerprints1_non_antibacterial = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules1_non_antibacterial]
    fingerprints2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules2]

    # Convert fingerprints to array
    fp_array1_antibacterial = np.zeros((len(fingerprints1_antibacterial), 1024))
    for i, fp in enumerate(fingerprints1_antibacterial):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array1_antibacterial[i] = arr

    fp_array1_non_antibacterial = np.zeros((len(fingerprints1_non_antibacterial), 1024))
    for i, fp in enumerate(fingerprints1_non_antibacterial):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array1_non_antibacterial[i] = arr

    fp_array2 = np.zeros((len(fingerprints2), 1024))
    for i, fp in enumerate(fingerprints2):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array2[i] = arr

    # Combine the fingerprints for dimensionality reduction
    combined_fp_array = np.vstack([fp_array1_antibacterial, fp_array1_non_antibacterial, fp_array2])

    # Define dimensionality reduction processes
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    for reduction_method in reduction_methods:
        # Perform dimensionality reduction
        reduced_data = perform_dimensionality_reduction(combined_fp_array, reduction_method)

        # Split the reduced data
        reduced_data1_antibacterial = reduced_data[:len(fp_array1_antibacterial)]
        reduced_data1_non_antibacterial = reduced_data[len(fp_array1_antibacterial):len(fp_array1_antibacterial) + len(fp_array1_non_antibacterial)]
        reduced_data2 = reduced_data[len(fp_array1_antibacterial) + len(fp_array1_non_antibacterial):]

        # Plot antibacterial vs. non-antibacterial molecules
        plt.figure(figsize=(12, 6))

        plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1],
                    label=f'New Molecules ({len(smiles_list2)})', color='black', alpha=0.6)

        plt.scatter(reduced_data1_non_antibacterial[:, 0], reduced_data1_non_antibacterial[:, 1],
                    label=f'Non-Antibacterial ({len(smiles_list1_non_antibacterial)})', color='red', alpha=0.6)

        plt.scatter(reduced_data1_antibacterial[:, 0], reduced_data1_antibacterial[:, 1],
                    label=f'Antibacterial ({len(smiles_list1_antibacterial)})', color='blue', alpha=0.6)

        plt.xlabel(f'{reduction_method} Component 1')
        plt.ylabel(f'{reduction_method} Component 2')
        plt.title(f'Antibacterial vs Non-Antibacterial vs New Molecules with {reduction_method}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, f"antibacterial_vs_nonantibacterial_vs_new_{reduction_method}.png"), bbox_inches='tight')
        plt.close()

        # Calculate QED for new molecules
        qed_scores = calculate_qed(smiles_list2)
        data2['QED'] = qed_scores

        # Save data2 with QED scores to a new file
        qed_output_file = os.path.join(output_dir, f"new_molecules_with_qed_{reduction_method}.csv")
        data2.to_csv(qed_output_file, index=False)

        # Plot QED values for new molecules
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(data2)), data2['QED'], color='skyblue')
        plt.xlabel('Molecule ID')
        plt.ylabel('QED Value')
        plt.title('QED Values for New Molecules')
        plt.xticks(range(len(data2)), data2['ID'], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"qed_values_bar_plot_{reduction_method}.png"))
        plt.close()

if __name__ == "__main__":
    args = cli()
    main(args.input1, args.input2, args.output)
