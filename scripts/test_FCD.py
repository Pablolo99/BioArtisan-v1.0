import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.linalg import sqrtm

# Function to read molecules from CSV or TSV
def read_molecules(csv_file, smiles_column, delimiter=','):
    df = pd.read_csv(csv_file, delimiter=delimiter)
    smiles_list = df[smiles_column].tolist()
    return smiles_list

# Function to convert SMILES to RDKit molecule objects
def smiles_to_mols(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    return [mol for mol in mols if mol is not None]

# Function to convert molecules to feature vectors (fingerprints)
def mols_to_feature_vectors(mols):
    fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(np.array(fp))
    return np.array(fps)

# Function to generate random probabilities for antibacterial activity
def random_model_predict_proba(feature_vectors):
    # Generate random probabilities between 0 and 1
    return np.random.rand(len(feature_vectors))

# Function to calculate the Fréchet Distance between two distributions
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Regularization to ensure positive semi-definite covariance matrices
    epsilon = 1e-6
    sigma1_reg = sigma1 + epsilon * np.eye(sigma1.shape[0])
    sigma2_reg = sigma2 + epsilon * np.eye(sigma2.shape[0])
    return np.sum(diff**2) + np.trace(sigma1_reg + sigma2_reg - 2 * sqrtm(sigma1_reg.dot(sigma2_reg), disp=False))

# Read molecule sets
antibacterial_smiles = [
    "Cc1cc(O)cc(O)c1C(=O)OC(C)Cc1cc(O)cc(O)c1C(=O)OC(C)Cc1cc(O)cc(O)c1C(=O)OC(C)Cc1cc(O)cc(O)c1C(=O)O",
    "Cc1cc(O)c2c(c1)C(=O)c1cccc(O)c1C2=O",
    "COc1c(C)c2c(c(O)c1CC=C(C)CCC(=O)OCCN1CCOCC1)C(=O)OC2",
    "COC1CC2CCC(C)C(O)(O2)C(=O)C(=O)N2CCCCC2C(=O)OC(C(C)CC2CCC(O)C(OC)C2)CC(=O)C(C)C=C(C)C(O)C(OC)C(=O)C(C)CC(C)C=CC=CC=C1C",
    "Cc1cc(O)cc(Oc2cc(C)cc(O)c2O)c1"
]

predicted_smiles = [
    "CCC1C(=O)C(CC)C23OC(S)C(OC)C(=O)C(CCCl)CC(CCCl)C(=O)C2OC1OC(C)C3(C=O)CC",
    "CCC1C(=O)C2OC(C3OC(S)C(C)C3=O)CC3OC4CC5(C)C(=O)C(N)C5(C)OC42C1(O)C3OC",
    "CCC1OC2(C1CC)C1OC3C4CC(N)C(=O)C(OC)C(=O)C(OC)C(=O)C(C)C(S)OC32C1O4",
    "CCC1OC2C34OC5C(O)C(=O)C3(C)C23OC(S)C(CC)C(=O)C(O)C(=O)C(OC)C(OC14)C(C)C(=O)C53OC",
    "CCC1C(=O)C(CCCl)C(=O)C(OC)C(=O)C(C)C(S)OC2(CC)C3OC1C(C)CC2(CC)C3=O"
]

# Convert SMILES to RDKit molecule objects
mols_antibacterial = smiles_to_mols(antibacterial_smiles)
mols_predicted = smiles_to_mols(predicted_smiles)

# Calculate feature vectors (Morgan fingerprints)
feature_vectors_antibacterial = mols_to_feature_vectors(mols_antibacterial)
feature_vectors_predicted = mols_to_feature_vectors(mols_predicted)

# Mock random predictions for antibacterial activity
antibacterial_probabilities = random_model_predict_proba(feature_vectors_antibacterial)

# Calculate mean and covariance of feature vectors
mu_antibacterial, sigma_antibacterial = np.mean(feature_vectors_antibacterial, axis=0), np.cov(feature_vectors_antibacterial, rowvar=False)
mu_predicted, sigma_predicted = np.mean(feature_vectors_predicted, axis=0), np.cov(feature_vectors_predicted, rowvar=False)

# Print some debug info
print("Mean vector 1:", mu_antibacterial.shape, mu_antibacterial[:5])
print("Mean vector 2:", mu_predicted.shape, mu_predicted[:5])
print("Covariance matrix 1 shape:", sigma_antibacterial.shape)
print("Covariance matrix 2 shape:", sigma_predicted.shape)

# Ensure covariance matrices are positive semi-definite
def is_positive_semi_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

print("Is sigma1 positive semi-definite?", is_positive_semi_definite(sigma_antibacterial))
print("Is sigma2 positive semi-definite?", is_positive_semi_definite(sigma_predicted))

# Calculate Fréchet ChemNet Distance
fcd = calculate_frechet_distance(mu_antibacterial, sigma_antibacterial, mu_predicted, sigma_predicted)
print("Fréchet ChemNet Distance:", fcd)
