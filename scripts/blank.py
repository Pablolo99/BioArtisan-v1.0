import pandas as pd

# Leer el archivo CSV original
archivo_original = "C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/clustering/PK_const/cluster_info_ConstrainedKMeans.csv"  # Reemplaza con la ruta a tu archivo
df = pd.read_csv(archivo_original)

# Crear una lista de diccionarios para almacenar las nuevas filas
nueva_lista = []

# Iterar sobre las filas del DataFrame original
for index, row in df.iterrows():
    # Obtener los valores necesarios
    smiles_id = row['mol_id']
    smiles = row['smiles']
    group_id = row['cluster']

    # Crear un diccionario con el formato deseado
    nueva_fila = {
        'smiles_id': smiles_id,
        'group_id': group_id,
        'smiles': smiles
    }

    # Añadir el diccionario a la lista
    nueva_lista.append(nueva_fila)

# Crear un nuevo DataFrame con las nuevas filas
nuevo_df = pd.DataFrame(nueva_lista)

# Guardar el nuevo DataFrame en un archivo CSV
nuevo_archivo_csv = "C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/clustering/PK_const/cluster_info_ConstrainedKMeans_clean.csv"  # Nombre del nuevo archivo
nuevo_df.to_csv(nuevo_archivo_csv, index=False)

print(f"Archivo guardado en {nuevo_archivo_csv}")