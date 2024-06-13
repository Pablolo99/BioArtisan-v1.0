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

def plot_qed_values(csv_file, output_file, top_n=10):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Sort the DataFrame by QED values in descending order
    df = df.sort_values(by='QED', ascending=False)

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df.index, df['QED'], color='skyblue')

    # Get the top N molecules
    top_n_df = df.head(top_n)

    # Print molecule IDs over each bar for the top N molecules
    for i, (idx, qed) in enumerate(zip(top_n_df.index, top_n_df['QED'])):
        plt.text(idx, qed, str(df['ID'][idx]), ha='center', va='bottom', fontsize=8)

    # Set the labels and title
    plt.xlabel('Molecule ID')
    plt.ylabel('QED Value')
    plt.title(f'QED Values of Molecules')

    # Set x-axis ticks to 1, 5, 10, 15, 20...
    plt.xticks(ticks=range(0, len(df), 9), labels=range(1, len(df) + 1, 9))

    # Save the plot
    plt.savefig(output_file)
    plt.close()

    # Save the top N molecules to a new file
    top_n_output_file = os.path.splitext(output_file)[0] + f"_top{top_n}.csv"
    top_n_df.to_csv(top_n_output_file, index=False)


def calculate_validity(smiles_list):
    valid_molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    validity = len(valid_molecules) / len(smiles_list)
    return validity, valid_molecules


def calculate_uniqueness(smiles_list):
    unique_smiles = set(smiles_list)
    uniqueness = len(unique_smiles) / len(smiles_list)
    return uniqueness


def calculate_novelty(new_smiles, original_smiles):
    original_smiles_set = set(original_smiles)
    novel_smiles = [smiles for smiles in new_smiles if smiles not in original_smiles_set]
    novelty = len(novel_smiles) / len(new_smiles)
    return novelty


def calculate_kl_divergence(p, q, epsilon=1e-10):
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    return entropy(p, q)


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

    fingerprints1_antibacterial = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in
                                   molecules1_antibacterial]
    fingerprints1_non_antibacterial = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in
                                       molecules1_non_antibacterial]
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
        reduced_data1_non_antibacterial = reduced_data[
                                           len(fp_array1_antibacterial):len(fp_array1_antibacterial) + len(
                                               fp_array1_non_antibacterial)]
        reduced_data2 = reduced_data[
                        len(fp_array1_antibacterial) + len(fp_array1_non_antibacterial):]

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
        plt.savefig(os.path.join(output_dir, f"antibacterial_vs_nonantibacterial_vs_new_{reduction_method}.png"),
                    bbox_inches='tight')
        plt.close()

    # Calculate Validity
    validity_new, valid_molecules_new = calculate_validity(smiles_list2)

    # Calculate Uniqueness
    uniqueness_new = calculate_uniqueness(smiles_list2)

    # Calculate Novelty
    novelty_new = calculate_novelty(smiles_list2,
                                    smiles_list1_antibacterial + smiles_list1_non_antibacterial)

    # Calculate KL Divergence
    kl_divergence_antibacterial = calculate_kl_divergence(
        np.histogram(fp_array1_antibacterial, bins=50, density=True)[0],
        np.histogram(fp_array2, bins=50, density=True)[0]
    )

    kl_divergence_non_antibacterial = calculate_kl_divergence(
        np.histogram(fp_array1_non_antibacterial, bins=50, density=True)[0],
        np.histogram(fp_array2, bins=50, density=True)[0]
    )

    # Print or save the calculated metrics
    metrics_output_file = os.path.join(output_dir, f"metrics.txt")
    with open(metrics_output_file, 'w') as f:
        f.write(f"Validity of New Molecules: {validity_new:.2f}\n")
        f.write(f"Uniqueness of New Molecules: {uniqueness_new:.2f}\n")
        f.write(f"Novelty of New Molecules: {novelty_new:.2f}\n")
        f.write(f"KL Divergence (Antibacterial): {kl_divergence_antibacterial:.10f}\n")
        f.write(f"KL Divergence (Non-Antibacterial): {kl_divergence_non_antibacterial:.10f}\n")

    # Calculate QED for new molecules
    qed_scores = calculate_qed(smiles_list2)
    data2['QED'] = qed_scores

    # Save data2 with QED scores to a new file
    qed_output_file = os.path.join(output_dir, f"new_molecules_with_qed.csv")
    data2.to_csv(qed_output_file, index=False)

    #plot QED
    out_path_plot = os.path.join(output_dir, f"new_molecules_qed.png")
    plot_qed_values(qed_output_file,out_path_plot)

if __name__ == "__main__":
    args = cli()
    main(args.input1, args.input2, args.output)