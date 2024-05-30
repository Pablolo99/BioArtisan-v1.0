import pandas as pd

# Read the CSV file
df = pd.read_csv('C:/Users/pablo/Downloads/antibacterial_NRP.csv')

# Fill NaN values in the 'antibacterial' column with 0
df['antibacterial'] = df['antibacterial'].fillna(0)

# Convert the 'antibacterial' column to integers
df['antibacterial'] = df['antibacterial'].astype(int)

# Replace values greater than 1 with 1
df.loc[df['antibacterial'] > 1, 'antibacterial'] = 1

# Save the modified DataFrame back to the same CSV file
df.to_csv('C:/Users/pablo/Downloads/antibacterial_NRP.csv', index=False)
