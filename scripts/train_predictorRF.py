#!/usr/bin/env pyton3
"""
Description:    Train a predictor to predict the activity of a molecule.
Usage:          python train_predictorRF.py -i data/train.csv -o output/ [OPTIONS]
"""
import argparse 
import joblib 
import typing as ty

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def cli() -> argparse.Namespace:
    """
    Command Line Interface
    
    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Training data in csv format, with individual lines formatted as 'id,smiles,antibacterial,Polyketide'.")
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

            # parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            fp = mol_to_fingerprint(mol, 2, 2048)
            data = np.vstack((data, fp)) if data.size else fp

            # parse labels
            labels = np.append(labels, int(label))

    return data, labels

def main() -> None:
    # turn RDKit warnings off
    Chem.rdBase.DisableLog("rdApp.*")

    # parse command line arguments
    args = cli()

    # parse data
    X, y = parse_data(args.input, args.h)

    # print dimensionality of data
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    # define parameter grid for RF
    param_grid_rf = {
        'max_depth': [5, 20, 50, 100],
        'n_estimators': [100, 500, 800, 1000]
    }

    # initialize RF classifier
    rf = RandomForestClassifier()

    # perform grid search with 5-fold cross-validation
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5)
    grid_search_rf.fit(X, y)

    # get the best parameters and score
    best_params_rf = grid_search_rf.best_params_
    best_score_rf = grid_search_rf.best_score_

    print("Best parameters for Random Forest:", best_params_rf)
    print("Best score for Random Forest:", best_score_rf)

    # save the best RF model
    best_model_rf = grid_search_rf.best_estimator_
    output_path_rf = f"{args.output}/model_RF.pkl"
    print(f"Saving best Random Forest model to {output_path_rf}...")
    joblib.dump(best_model_rf, output_path_rf)

    exit(0)

if __name__ == "__main__":
    main()