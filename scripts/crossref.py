#!/usr/bin/env pyton3

"""
This script is developed in order to crossreference molecular information of two csv files.
It returns a csv file with additional information of what kind of molecule we have.
In order to do so, it compares the fingerprints of molecules of two information files.
"""
import os
import pandas as pd
import argparse
import csv
import numpy as np
import typing as ty

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


def cli() -> argparse.Namespace:
    """
    Command Line Interface

    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", type=str, required=True,
                        help="Training data in csv format, with individual lines formatted as 'id,smiles,0/1'.")
    parser.add_argument("-i2", "--input2", type=str, required=True,
                        help="Training data in tsv format, with individual\
                lines formatted as 'num,coconut_id,smiles,prediction_pathway'.")
    parser.add_argument("-np", "--natural_product", type=str, required=True,
                        help="Natural product class specified")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output data in csv format, with individual lines formatted as 'mol_id,smiles,antibacterial,{np}' ")

    return parser.parse_args()
def tsv_to_csv(tsv_file):
    """
    This function converts a tsv format file to a csv format file
    :param tsv_file: str, PATH of the original tsv file
    :param csv_file: str, PATH of the created csv file
    """
    base_name, _ = os.path.splitext(os.path.basename(tsv_file))
    csv_file = os.path.join(os.path.dirname(tsv_file), f"{base_name}.csv")

    if tsv_file.lower().endswith('.csv'):
        print(f"{tsv_file} is already a CSV file. No conversion needed.")
        return tsv_file

    elif tsv_file.lower().endswith('.tsv'):
        df = pd.read_csv(tsv_file, sep='\t')
        df.to_csv(csv_file, index=False)
        print(f"Conversion successful: {tsv_file} -> {csv_file}")
    return csv_file


def filter_np(csv_file,class_np):
    """
    This function creates a csv file that only contains lines that have the word class
    :param csv_file: str, PATH of the original csv file
    :param class_np: str, word to look for in each line of csv file
    :return: filter_csv_file : str, PATH of the filtered csv file
    """
    full_path, file_name = os.path.split(csv_file)
    base_name, _ = os.path.splitext(file_name)
    full_base_name = os.path.join(full_path, base_name)

    output_csv = f'{full_base_name}_filtered_{class_np}_output.csv'
    #check if the output file already exists
    if os.path.exists(output_csv):
        print(f"Filtered csv file already exists: {output_csv}. No filtering needed")
        return output_csv

    df = pd.read_csv(csv_file)
    filtered_df = df[df.apply(lambda row: class_np in row.to_string(), axis=1)]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtering successful: {csv_file} -> {output_csv}")
    return output_csv


def calculate_morgan_fingerprint(smiles, radius=2, num_bits=2048):
    """
    This function calculates Morgan fingerprint for a molecule in SMILES format
    :param smiles: str, SMILES format of the molecule
    :param radius: int, radius specification for the fingerprint (2).
    :param num_bits: int, number of bits used (2048)
    :return: RDKit Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Failed to parse SMILES.")

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, num_bits)
    return fingerprint

def calculate_tanimoto(fp1, fp2):
    """
    This function calculates the Tanimoto similarity index between 2 fingerprints
    :param fp1: RDKit fingerprint for the first molecule
    :param fp2: RDKit fingerprint for the second molecule
    :return: float, Tanimoto index
    """
    tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    return tanimoto_similarity


def add_np_column(file_path1, file_path2, output_path, np):
    """
    This function adds a new column to a df with crossreferenced information.
    The original df is a csv file of the donphan db. The information added is
    within the second file, a csv file original from coconut db.
    :param file_path1: str, path of the donphan db csv file
    :param file_path2: str, path of the coconut prediction file csv format
    :param output_path: str, path of the resulting csv file
    :patam np: str, name of the class of natural product
    """
    df1 = pd.read_csv(file_path1)

    #extract fingerprints for the second file
    df2 = pd.read_csv(file_path2)
    smiles_list2 = df2['smiles'].tolist()
    fingerprints_dict = {}

    for smiles2 in smiles_list2:
        if smiles2 is not None:
            fp2 = calculate_morgan_fingerprint(smiles2)
            fingerprints_dict[smiles2] = fp2

    for i, smiles1 in enumerate(df1['smiles']):
        if smiles1 is not None:
            fp1 = calculate_morgan_fingerprint(smiles1)
        for smiles2, fp2 in fingerprints_dict.items():
            # calculate Tanimoto similarity
            tanimoto_similarity = calculate_tanimoto(fp2, fp1)

            if tanimoto_similarity > 0.99:
                df1.loc[i, np] = '1'
                break

    #drop all the lines that are not recognized
    df1 = df1.dropna(subset=[np])

    #drop all the lines that have not 1 in the last column
    df1_filtered = df1[df1[np] == '1']

    df1_filtered.to_csv(output_path, index=False)


def main():
    # turn RDKit warnings off
    Chem.rdBase.DisableLog("rdApp.*")

    #parse command line argument
    args = cli()
    file1 = args.input1
    file2 = args.input2
    np = args.natural_product
    output_path = args.output

    #convert tsv file to csv
    file2_csv = tsv_to_csv(file2)

    #filter csv by np
    filtered_csv = filter_np(file2_csv,np)

    #add natural product column to the information of file1
    add_np_column(file1, filtered_csv, output_path, np)


if __name__ == "__main__":
    main()