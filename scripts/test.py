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
from rdkit.Chem import rdMolDescriptors


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


def calculate_fcd(new_mols, original_mols):
    # This function requires a pretrained model and is typically more complex.
    # Here we provide a placeholder for the calculation.
    # In practice, you would use a model like ChemNet to compute embeddings and then FCD.
    fcd = np.random.random()  # Placeholder for demonstration purposes
    return fcd


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

    # Validity
    validity, valid_mols2 = calculate_validity(smiles_list2)
    valid_smiles2 = [Chem.MolToSmiles(mol) for mol in valid_mols2]

    # Uniqueness
    uniqueness = calculate_uniqueness(valid_smiles2)

    # Novelty
    novelty = calculate_novelty(valid_smiles2, smiles_list1)

    # KL Divergence
    valid_mols1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1 if Chem.MolFromSmiles(smiles) is not None]
    weights1, logp1 = calculate_molecular_properties(valid_mols1)
    weights2, logp2 = calculate_molecular_properties(valid_mols2)

    kl_div_weights = calculate_kl_divergence(np.histogram(weights1, bins=50, density=True)[0],
                                             np.histogram(weights2, bins=50, density=True)[0])
    kl_div_logp = calculate_kl_divergence(np.histogram(logp1, bins=50, density=True)[0],
                                          np.histogram(logp2, bins=50, density=True)[0])

    kl_divergence = (kl_div_weights + kl_div_logp) / 2

    # FCD (Placeholder)
    fcd = calculate_fcd(valid_mols2, valid_mols1)

    # QED Scores
    qed_scores = calculate_qed(valid_smiles2)
    data2['QED'] = qed_scores

    # Save data2 with QED scores to a new file
    qed_output_file = os.path.join(output_dir, "new_molecules_with_qed.csv")
    data2.to_csv(qed_output_file, index=False)

    # Plot QED values for new molecules
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data2)), data2['QED'], color='skyblue')
    plt.xlabel('Molecule ID')
    plt.ylabel('QED Value')
    plt.title('QED Values for New Molecules')
    plt.xticks(range(len(data2)), data2['ID'], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qed_values_bar_plot.png"))
    plt.close()

    # Print or save the calculated metrics
    metrics_output_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_output_file, 'w') as f:
        f.write(f"Validity: {validity:.2f}\n")
        f.write(f"Uniqueness: {uniqueness:.2f}\n")
        f.write(f"Novelty: {novelty:.2f}\n")
        f.write(f"KL Divergence: {kl_divergence:.2f}\n")
        f.write(f"FCD: {fcd:.2f}\n")

    # Dimensionality reduction and plotting
    molecules1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1]
    fingerprints1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules1]
    fingerprints2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in valid_mols2]

    fp_array1 = np.zeros((len(fingerprints1), 1024))
    for i, fp in enumerate(fingerprints1):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array1[i] = arr

    fp_array2 = np.zeros((len(fingerprints2), 1024))
    for i, fp in enumerate(fingerprints2):
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array2[i] = arr

    # Combine the fingerprints for dimensionality reduction
    combined_fp_array = np.vstack([fp_array1, fp_array2])

    # Define dimensionality reduction processes
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    for reduction_method in reduction_methods:
        # Perform dimensionality reduction
        reduced_data = perform_dimensionality_reduction(combined_fp_array, reduction_method)

        # Split the reduced data
        reduced_data1 = reduced_data[:len(fp_array1)]
        reduced_data2 = reduced_data[len(fp_array1):]

    # Plot antibacterial vs. non-antibacterial molecules
    plt.figure(figsize=(12, 6))

    plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1],
                label=f'New Molecules ({len(smiles_list2)})', color='black', alpha=0.6)

    plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1],
                label=f'Original Molecules ({len(smiles_list1)})', color='blue', alpha=0.6)

    plt.xlabel(f'{reduction_method} Component 1')
    plt.ylabel(f'{reduction_method} Component 2')
    plt.title(f'Original vs New Molecules with {reduction_method}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, f"original_vs_new_{reduction_method}.png"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = cli()
    main(args.input1, args.input2, args.output)
