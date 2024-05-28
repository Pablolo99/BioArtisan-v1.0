import csv


def remove_last_column_and_modify_antibacterial(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read the header
        header = next(reader)
        writer.writerow(header[:-1])  # Write all columns except the last one

        for row in reader:
            if len(row) > 1:  # Check to make sure the row isn't empty
                # Modify the 'antibacterial' column
                antibacterial_value = row[2]
                if antibacterial_value == '0':
                    row[2] = ''  # Replace '0' with an empty string
                writer.writerow(row[:-1])  # Write all columns except the last one


if __name__ == "__main__":
    input_file = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/antibacterial_PK.csv'
    output_file = 'C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/data/antibacterial_PK_clean.csv'
    remove_last_column_and_modify_antibacterial(input_file, output_file)
