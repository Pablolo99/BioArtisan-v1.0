import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap
from molscore.metrics import Validity, Uniqueness, Novelty, KLDivergence, FCD

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", type=str, required=True, help="Input data file 1 in CSV format.")
    parser.add_argument("-i2", "--input2", type=str, required=True, help="Input data file 2 in TSV format.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory to store results.")
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
    return dim_red.fit_transform(data)

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
    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2, delimiter='\t')

    smiles_list1 = data1['smiles'].tolist()
    smiles_list2 = data2['SMILES'].tolist()

    validity_metric = Validity()
    uniqueness_metric = Uniqueness()
    novelty_metric = Novelty(smiles_list1)
    kl_divergence_metric = KLDivergence()
    fcd_metric = FCD()

    validity = validity_metric(smiles_list2)
    uniqueness = uniqueness_metric(smiles_list2)
    novelty = novelty_metric(smiles_list2)
    kl_divergence = kl_divergence_metric(smiles_list1, smiles_list2)
    fcd = fcd_metric(smiles_list1, smiles_list2)

    qed_scores = calculate_qed(smiles_list2)
    data2['QED'] = qed_scores

    qed_output_file = os.path.join(output_dir, "new_molecules_with_qed.csv")
    data2.to_csv(qed_output_file, index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data2)), data2['QED'], color='skyblue')
    plt.xlabel('Molecules')
    plt.ylabel('QED Value')
    plt.title('QED Values for New Molecules')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qed_values_bar_plot.png"))
    plt.close()

    metrics_output_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_output_file, 'w') as f:
        f.write(f"Validity: {validity:.2f}\n")
        f.write(f"Uniqueness: {uniqueness:.2f}\n")
        f.write(f"Novelty: {novelty:.2f}\n")
        f.write(f"KL Divergence: {kl_divergence:.2f}\n")
        f.write(f"FCD: {fcd:.2f}\n")

    molecules1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list1]
    fingerprints1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules1]
    fingerprints2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in [Chem.MolFromSmiles(smiles) for smiles in smiles_list2]]

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

    combined_fp_array = np.vstack([fp_array1, fp_array2])
    reduction_methods = ["PCA", "MDS", "t-SNE", "UMAP"]

    for reduction_method in reduction_methods:
        reduced_data = perform_dimensionality_reduction(combined_fp_array, reduction_method)

        reduced_data1 = reduced_data[:len(fp_array1)]
        reduced_data2 = reduced_data[len(fp_array1):]

        plt.figure(figsize=(12, 6))
        plt.scatter(reduced_data1[:, 0], reduced_data1[:, 1], label='Original Molecules', color='blue', alpha=0.6)
        plt.scatter(reduced_data2[:, 0], reduced_data2[:, 1], label='New Molecules', color='black', alpha=0.6)
        plt.xlabel(f'{reduction_method} Component 1')
        plt.ylabel(f'{reduction_method} Component 2')
        plt.title(f'Original vs New Molecules with {reduction_method}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        plt.savefig(os.path.join(output_dir, f"original_vs_new_{reduction_method}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    args = cli()
    main(args.input1, args.input2, args.output)
