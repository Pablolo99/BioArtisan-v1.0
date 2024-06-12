from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

# Define the list of subunits
subunits = [
    '[S][C](=[O])[CH2][C](=[O])[OH]',
    '[S][C]([OH])=[CH][C](=[O])[OH]',
    '[S][C](=[O])[CH1]([CH3])[C](=[O])[OH]',
    '[S][C]([OH])=[C]([CH3])[C](=[O])[OH]',
    '[S][C](=[O])[C]([CH3])([CH3])[C](=[O])[OH]',
    '[S][C](=[O])[CH1]([CH2][CH3])[C](=[O])[OH]',
    '[S][C]([OH])=[C]([CH2][CH3])[C](=[O])[OH]',
    '[S][C](=[O])[CH1]([OH])[C](=[O])[OH]',
    '[S][C]([OH])=[C]([OH])[C](=[O])[OH]',
    '[S][C](=[O])[CH1]([CH2][OH])[C](=[O])[OH]',
    '[S][C]([OH])=[C]([CH2][OH])[C](=[O])[OH]',
    '[S][C](=[O])[C]([CH3])([OH])[C](=[O])[OH]',
    '[S][C](=[CH2])[CH2][C](=[O])[OH]',
    '[S][C](=[CH2])[CH1]([CH3])[C](=[O])[OH]',
    '[S][C](=[CH2])[CH1]([CH3])[C](=[O])[OH]',
    '[S][C](=[CH2])[C](=[O])[C](=[O])[OH]',
    '[S][C](=[O])[C](=[O])[C](=[O])[OH]',
    '[S][CH1]([OH])[CH2][C](=[O])[OH]',
    '[S][CH1]([OH])[CH1]([CH3])[C](=[O])[OH]',
    '[S][CH1]([OH])[C]([CH3])([CH3])[C](=[O])[OH]',
    '[S][CH1]([OH])[C]([CH2][CH3])[C](=[O])[OH]',
    '[S][CH1]([OH])[CH1]([OH])[C](=[O])[OH]',
    '[S][CH1]([OH])[CH1]([CH2][OH])[C](=[O])[OH]',
    '[S][CH1]([OH])[CH0]([CH3])([OH])[C](=[O])[OH]',
    '[S][CH0]([CH3])([OH])[CH2][C](=[O])[OH]',
    '[S][CH0]([CH3])([OH])[CH1]([CH3])[C](=[O])[OH]',
    '[S][CH0]([CH3])([OH])[CH1]([OH])[C](=[O])[OH]',
    '[S][CH]([OH])[C](=[O])[C](=[O])[OH]',
    '[S][CH]([OH])[CH]([NH2])[C](=[O])[OH]',
    '[S][CH1]=[CH1][C](=[O])[OH]',
    '[S][CH1]=[C]([CH2][CH3])[C](=[O])[OH]',
    '[S][C]([CH3])=[CH1][C](=[O])[OH]',
    '[S][CH1]=[C]([CH0]#[N])[C](=[O])[OH]',
    '[S][CH2][CH2][C](=[O])[OH]',
    '[S][CH2][CH1]([CH3])[C](=[O])[OH]',
    '[S][CH2][C]([CH3])([CH3])[C](=[O])[OH]',
    '[S][CH2][CH1]([CH2][CH3])[C](=[O])[OH]',
    '[S][CH2][CH]([OH])[C](=[O])[OH]',
    '[S][CH2][CH]([CH2][OH])[C](=[O])[OH]',
    '[S][CH2][C]([CH3])([OH])[C](=[O])[OH]',
    '[S][CH1]([CH3])[CH2][C](=[O])[OH]',
    '[S][CH1]([CH3])[CH1]([OH])[C](=[O])[OH]',
    '[S][CH2][C](=[O])[C](=[O])[OH]',
    '[S][CH1]=[C]([OH])[C](=[O])[OH]',
    '[S][CH2][C](=[CH2])[C](=[O])[OH]'
]

# Define the reaction SMARTS for both cases
reaction_with_double_bond = '[S:1][C:2].[OH]-[C](=[O])-[C:3]=[C:4][S]>>[C:2][C:3]=[C:4][S:1]'
reaction_without_double_bond = '[S:1][C:2].[OH]-[C](=[O])-[C:3][C:4][S]>>[C:2][C:3][C:4][S:1]'
rxn_with_double_bond = rdChemReactions.ReactionFromSmarts(reaction_with_double_bond)
rxn_without_double_bond = rdChemReactions.ReactionFromSmarts(reaction_without_double_bond)

results = []
no_products = []

for subunit_smiles in subunits:
    subunit = Chem.MolFromSmiles(subunit_smiles)
    reactants = (subunit, subunit)

    # Check for double bond presence
    double_bond_pattern = Chem.MolFromSmarts('[OH]-[C](=[O])-[C:3]=[C:4][S]')
    if subunit.HasSubstructMatch(double_bond_pattern):
        reaction = rxn_with_double_bond
    else:
        reaction = rxn_without_double_bond

    # Perform the reaction
    products = reaction.RunReactants(reactants)

    if products:
        # Save the results
        for product_set in products:
            for product in product_set:
                product_smiles = Chem.MolToSmiles(product)
                results.append((subunit_smiles, product_smiles))
    else:
        no_products.append(subunit_smiles)

# Print the subunits that produced products
print("Subunits that produced products:")
for subunit_smiles, product_smiles in results:
    print(subunit_smiles)
    print(product_smiles)

# Print the subunits that did not produce products
print("\nSubunits that did not produce products:")
for subunit_smiles in no_products:
    print(f"Reactant: {subunit_smiles}")
