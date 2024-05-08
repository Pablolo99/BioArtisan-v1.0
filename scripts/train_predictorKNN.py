#!/usr/bin/env pyton3
"""
Description:    Train a predictor to predict the activity of a molecule.
Usage:          python train_predictorSVM.py -i data/train.csv -o output/ [OPTIONS]
"""
import argparse
import joblib
import typing as ty

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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


def parse_data(path: str, header: bool) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Parse training data from csv file.

    :param str path: Path to csv file.
    :param bool header: Flag to indicate if csv file has header.
    :return: Tuple of training data and labels.
    :rtype: ty.Tuple[np.ndarray, np.ndarray]
    """
    data = np.array([])
    labels = np.array([])

    with open(path, "r") as file_open:

        if header:
            file_open.readline()

        for i, line in enumerate(file_open):
            _, smiles, label, type = line.strip().split(",")

            # Parse SMILES.
            mol = Chem.MolFromSmiles(smiles)
            fp = mol_to_fingerprint(mol, 2, 2048)
            data = np.vstack((data, fp)) if data.size else fp

            # Parse label.
            labels = np.append(labels, int(label))

    return data, labels


def main() -> None:
    # Turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.*")

    # Parse command line arguments.
    args = cli()

    # Parse data.
    X, y = parse_data(args.input, args.h)

    # Print dimensionality of data.
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    # Define parameter grid for KNN
    param_grid_knn = {
        'n_neighbors': [2, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Initialize KNN classifier
    knn = KNeighborsClassifier()

    # Perform grid search with 5-fold cross-validation
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5)
    grid_search_knn.fit(X, y)

    # Get the best parameters and score
    best_params_knn = grid_search_knn.best_params_
    best_score_knn = grid_search_knn.best_score_

    print("Best parameters for KNN:", best_params_knn)
    print("Best score for KNN:", best_score_knn)

    # Save the best KNN model
    best_model_knn = grid_search_knn.best_estimator_
    output_path_knn = f"{args.output}/model_KNN.pkl"
    print(f"Saving best KNN model to {output_path_knn}...")
    joblib.dump(best_model_knn, output_path_knn)

    exit(0)


if __name__ == "__main__":
    main()