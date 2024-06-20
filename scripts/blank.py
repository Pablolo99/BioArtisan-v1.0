# Read the file and find duplicate SMILES strings
file_path = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/output_PKtest_200'

# Dictionary to store SMILES and their corresponding line numbers
smiles_dict = {}

with open(file_path, 'r') as file:
    header = file.readline()  # Read the header line
    lines = file.readlines()  # Read the rest of the lines

    # Process each line to find duplicates
    for index, line in enumerate(lines, start=2):  # Start from 2 to account for the header line
        columns = line.strip().split('\t')
        if len(columns) < 3:
            continue
        smiles = columns[1]

        if smiles in smiles_dict:
            smiles_dict[smiles].append(index)
        else:
            smiles_dict[smiles] = [index]

# Print lines with duplicate SMILES
print("Lines with duplicate SMILES strings:")
for smiles, line_numbers in smiles_dict.items():
    if len(line_numbers) > 1:
        print(f"SMILES: {smiles}")
        for line_number in line_numbers:
            print(f"  Line {line_number}: {lines[line_number-2].strip()}")
