import pandas as pd

# Read the CSV file
df = pd.read_csv('C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/clustering/NRP/cluster_info_KMeans.csv')

# Modify the 'antibacterial' column
df['antibacterial'] = df['antibacterial'].apply(lambda x: 1 if x != 0 else 0)

# Save the modified DataFrame back to a new CSV file
df.to_csv('C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/clustering/NRP/cluster_info_KMeans.csv', index=False)