#!/usr/bin/env python3
"""
Description:    Train predictors to predict the activity of a molecule.
Usage:          python model_evaluation.py -i data/train.csv -o output/ [OPTIONS]
"""
import argparse
import os
import pandas as pd
import numpy as np
import itertools
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def cli() -> argparse.Namespace:
    """
    Command Line Interface

    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Training data in csv format, with individual lines formatted as 'id,smiles,antibacterial,Polyketide'.")
    parser.add_argument("--h", "--header", action="store_true", help="Flag to indicate if csv file has header.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output dir.")
    return parser.parse_args()


def mol_to_fingerprint(mol: Chem.Mol, radius: int, num_bits: int) -> np.ndarray:
    """
    Convert molecule to Morgan fingerprint.

    :param Chem.Mol mol: Molecule to convert.
    :param int radius: Radius of fingerprint.
    :param int num_bits: Number of bits in fingerprint.
    :return: Fingerprint of molecule.
    :rtype: np.ndarray
    """
    bit_fingerprint = np.zeros((0,), dtype=int)
    morgan_bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, num_bits)
    DataStructs.ConvertToNumpyArray(morgan_bit_vector, bit_fingerprint)
    return bit_fingerprint


def parse_data(path: str, header: bool) -> dict:
    """
    Parse training data from csv file and return a dict of the clusters information.

    :param str path: Path to csv file.
    :param bool header: Flag to indicate if csv file has header.
    :return: Tuple of training data and labels.
    :rtype: dict[int, tuple[np.ndarray, np.ndarray]]
    """
    if header:
        data = pd.read_csv(path, header=0)
    else:
        data = pd.read_csv(path, header=None)
        data.columns = ["id", "smiles", "antibacterial", "Polyketide", "cluster"]

    clusters_dic = {}
    clusters = np.unique(data['cluster'])  # Unique cluster values

    for cluster in clusters:
        cluster_data = data[data['cluster'] == cluster]
        smiles_list = cluster_data['smiles']
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in molecules]
        fps = np.array([np.zeros(0)] if len(fingerprints) == 0 else fingerprints)
        labels = cluster_data['antibacterial']
        clusters_dic[cluster] = (fps, labels)

    return clusters_dic


def save_confusion_matrix(y_true, y_pred, model_name, param_key, output_dir) -> None:
    """
    Save confusion matrix to a text file and plot heatmap.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param str model_name: Name of the model.
    :param str param_key: Key representing the model's hyperparameters.
    :param str output_dir: Output directory.
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    tn, fp, fn, tp = cm.ravel()
    output_path = os.path.join(output_dir, f"{model_name}_{param_key}_confusion_matrix.txt")
    with open(output_path, 'w') as f:
        f.write(f"Confusion Matrix for {model_name} with hyperparameters {param_key}:\n")
        f.write(f"TN: {tn}  FP: {fp}\n")
        f.write(f"FN: {fn}  TP: {tp}\n")
    print(f"Saved confusion matrix to {output_path}")

    # Plot heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name} with hyperparameters {param_key}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    heatmap_path = os.path.join(output_dir, f"{model_name}_{param_key}_confusion_matrix_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved confusion matrix heatmap to {heatmap_path}")


def train_models(clusters_dic: dict, output_dir: str) -> dict:
    """
    Train and evaluate models for different algorithms for each cluster.

    :param dict clusters_dic: Dictionary containing cluster information.
    :param str output_dir: Output directory to store results and models.
    :return: Dictionary containing evaluation results.
    :rtype: dict
    """
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "GBM": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}

    # define parameter grids for grid search
    param_grids = {
        "RandomForest": {
            'max_depth': [5, 20, 50, 100],
            'n_estimators': [100, 500, 800, 1000]
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        },
        "GBM": {
            'loss': ['log_loss', 'exponential'],
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 10]
        },
        "KNN": {
            'n_neighbors': [2, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 50]
        }
    }

    for model_name in models.keys():
        model = models[model_name]
        param_grid = param_grids[model_name]

        results[model_name] = {}

        param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

        for params in param_combinations:
            key = '_'.join([f"{param}_{value}" for param, value in params.items()])
            results[model_name][key] = []

            for test_cluster, (test_X, test_y) in clusters_dic.items():
                train_X = []
                train_y = []

                for train_cluster, (X, y) in clusters_dic.items():
                    if train_cluster != test_cluster:
                        train_X.extend(X)
                        train_y.extend(y)

                # Ensure all elements in train_X are numpy arrays of the same shape
                train_X = np.array([np.array(fp) for fp in train_X])
                train_y = np.array(train_y)

                model.set_params(**params)
                model.fit(train_X, train_y)

                y_pred = model.predict(test_X)
                accuracy = accuracy_score(test_y, y_pred)
                f1 = f1_score(test_y, y_pred, average='macro')  # Calculate macro-average F1 score

                results[model_name][key].append((accuracy, f1))

            avg_accuracy = np.mean([score[0] for score in results[model_name][key]])
            avg_f1 = np.mean([score[1] for score in results[model_name][key]])
            print(f"Model: {model_name}, Hyperparameters: {key}, Average Accuracy: {avg_accuracy}, Average F1 Score: {avg_f1}")

        # Get the test data for the current model to calculate the confusion matrix
        for test_cluster, (test_X, test_y) in clusters_dic.items():
            train_X = np.concatenate([X for cluster, (X, y) in clusters_dic.items() if cluster != test_cluster])
            train_y = np.concatenate([y for cluster, (X, y) in clusters_dic.items() if cluster != test_cluster])
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)

            # Save the confusion matrix for the current model
            save_confusion_matrix(test_y, y_pred, model_name, key, output_dir)

    return results

def plot_results(results: dict, output_dir: str) -> None:
    """
    Plot the results of the model training.

    :param dict results: Dictionary containing evaluation results.
    :param str output_dir: Output directory to save plots.
    """
    for model_name, model_params in results.items():
        accuracies = [np.mean([score[0] for score in param_values]) for param_values in model_params.values()]
        f1_scores = [np.mean([score[1] for score in param_values]) for param_values in model_params.values()]
        param_keys = list(model_params.keys())

        plt.figure(figsize=(10, 5))
        plt.plot(param_keys, accuracies, label='Accuracy', marker='o')
        plt.plot(param_keys, f1_scores, label='F1 Score', marker='o')
        plt.xlabel('Parameter Combination')
        plt.ylabel('Score')
        plt.title(f'{model_name} Performance')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'{model_name}_performance.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")


def main() -> None:
    # turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.*")

    # parse command line arguments.
    args = cli()

    # obtain dict of clusters
    clusters_dic = parse_data(args.input, args.h)

    # train the models
    results_models = train_models(clusters_dic, args.output)

    # create the plot
    plot_results(results_models, args.output)


if __name__ == "__main__":
    main()
