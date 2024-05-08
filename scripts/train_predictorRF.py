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
from sklearn.model_selection import cross_val_score


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

    # Pick best settings random forest model with 5-fold cross validation.
    max_depth = [5, 20, 50, 100]
    estimators = [100, 500, 1000]

    best_score = 0
    best_depth = 0
    best_estimators = 0

    for depth in max_depth:
        for estimator in estimators:
            print(f"Training random forest with max_depth={depth} and n_estimators={estimator}...")
            model = RandomForestClassifier(max_depth=depth, n_estimators=estimator, n_jobs=-1)
            scores = cross_val_score(model, X, y, cv=5)
            score = np.mean(scores)
            print(f"Score: {score}")

            if score > best_score:
                best_score = score
                best_depth = depth
                best_estimators = estimator

    print(f"Best score: {best_score}")
    print(f"Best depth: {best_depth}")
    print(f"Best estimators: {best_estimators}")

    # Train model with best settings.
    print(f"Training random forest with max_depth={best_depth} and n_estimators={best_estimators}...")
    model = RandomForestClassifier(max_depth=best_depth, n_estimators=best_estimators, n_jobs=-1)
    model.fit(X, y)

    # Save model.
    print(f"Saving model to {args.output}...")
    joblib.dump(model, f"{args.output}/model_RF.pkl")

    exit(0)

if __name__ == "__main__":
    main()