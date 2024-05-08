#!/usr/bin/env pyton3
"""
Description:    Train predictors to predict the activity of a molecule.
Usage:          python model_evaluation.py -i data/train.csv -o output/ [OPTIONS]
"""
import argparse
import typing as ty
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
    :rtype: dic[int] = ty.Tuple[np.ndarray, np.ndarray]
    """
    if header:
        data = pd.read_csv(path, header=0)
    else:
        data = pd.read_csv(path, header=None)

    clusters_dic = {}
    clusters = np.unique(data['cluster'])  # Unique cluster values

    for cluster in clusters:
        cluster_data = data[data['cluster'] == cluster]
        smiles_list = cluster_data['smiles']
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]
        fps = np.array([np.zeros(0)] if len(fingerprints) == 0 else fingerprints)
        labels = cluster_data['antibacterial']
        clusters_dic[cluster] = (fps, labels)

    return clusters_dic


def train_models(clusters_dic: dict) -> ty.List[dict]:
    """"""
    # train and evaluate models for different algorithms for each cluster
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        #"GBM": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = []

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
            'loss': ['deviance', 'exponential'],
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

    # divide in train and test clusters
    for test_cluster, (test_X, test_y) in clusters_dic.items():
        train_X = []
        train_y = []

        for cluster, (X, y) in clusters_dic.items():
            if cluster != test_cluster:
                train_X.extend(X)
                train_y.extend(y)

        train_X = np.array(train_X)
        train_y = np.array(train_y)

        # evaluate each model
        cluster_results = {'Test Cluster': test_cluster}

        #evaluate each model
        for model_name in models.keys():
            model = models[model_name]
            param_grid = param_grids[model_name]

            # perform grid search with 5-fold cross-validation
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
            grid_search.fit(train_X, train_y)

            best_model = grid_search.best_estimator_

            best_model.fit(train_X, train_y)
            y_pred = model.predict(test_X)
            accuracy = accuracy_score(test_y, y_pred)
            cluster_results[model_name] = accuracy

        results.append(cluster_results)

    return results




def main() -> None:
    # turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.*")

    # parse command line arguments.
    args = cli()

    # obtain dict of clusters
    clusters_dic = parse_data(args.input, args.h)

    # train the models
    results_models = train_models(clusters_dic)

    print(results_models)
"""
    # write results to csv file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames =
    return results
"""

if __name__ == "__main__":
    main()
