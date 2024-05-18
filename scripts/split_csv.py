import os
import pandas as pd

def split_csv(file_path, output_dir, chunk_size=200):
    # Lire le fichier CSV
    df = pd.read_csv(file_path)
    
    # Calculer le nombre de lignes par chunk
    total_size = os.path.getsize(file_path) / (1024 * 1024)  # Taille du fichier en MB
    total_rows = len(df)
    rows_per_chunk = int((chunk_size / total_size) * total_rows)
    
    # Diviser le DataFrame en morceaux
    for i, chunk in enumerate(range(0, total_rows, rows_per_chunk)):
        chunk_df = df.iloc[chunk:chunk + rows_per_chunk]
        chunk_file = os.path.join(output_dir, f'meteo_idaweb_chunk_{i}.csv')
        chunk_df.to_csv(chunk_file, index=False)
        print(f"Écrit {chunk_file}")

# Exemple d'utilisation
split_csv('C:/Users/Abdi/Desktop/data/raw/meteo_idaweb.csv', 'C:/Users/Abdi/Desktop/data/split')
