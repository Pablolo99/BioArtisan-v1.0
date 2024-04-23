from rdkit import Chem

# Define the SMILES string
smiles = "CCC(=O)C1(CCCl)C(=O)CC(=O)C(CCCl)C1(O)S"

# Convert SMILES string to RDKit molecule object
mol = Chem.MolFromSmiles(smiles)

# Define SMARTS pattern for carboxylic group
carboxylic_pattern = Chem.MolFromSmarts("C(=O)O")

# Check if the molecule contains the carboxylic group
if mol.HasSubstructMatch(carboxylic_pattern):
    print("The molecule contains a carboxylic group.")
else:
    print("The molecule does not contain a carboxylic group.")