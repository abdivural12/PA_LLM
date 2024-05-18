import pandas as pd

# Charger le fichier complet
df = pd.read_csv('C:/Users/Abdi/Desktop/data/raw/meteo_idaweb.csv')

# Définir la taille de chaque morceau (en nombre de lignes)
chunk_size = 500000  # Ajustez selon vos besoins

# Diviser le fichier en morceaux plus petits
for i, chunk in enumerate(range(0, len(df), chunk_size)):
    chunk_df = df.iloc[chunk:chunk + chunk_size]
    chunk_df.to_csv(f'C:/Users/Abdi/Desktop/data/raw/meteo_idaweb_chunk_{i}.csv', index=False)
