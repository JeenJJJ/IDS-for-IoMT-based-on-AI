import pandas as pd
from ctgan import CTGAN
import numpy as np

# ======================
# 1. CHARGER ÉCHANTILLON
# ======================
print("Chargement des données...")
df = pd.read_csv('data/wustl-ehms-2020_with_attacks_categories.csv', nrows=5000)

print(f"Distribution des attaques :")
print(df['Attack Category'].value_counts())

# ======================
# 2. PRÉPARER LES DONNÉES
# ======================
# Colonnes numériques à garder
numeric_cols = [
    'Sport', 'Dport', 'SrcBytes', 'DstBytes', 'SrcLoad', 'DstLoad',
    'Loss', 'pLoss', 'Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA',
    'Heart_rate', 'Resp_Rate', 'ST'
]

# Garder les numériques + la colonne catégorielle
df_ctgan = df[numeric_cols + ['Attack Category']].copy()

# Nettoyer les valeurs
df_ctgan = df_ctgan.replace([np.inf, -np.inf], np.nan)
df_ctgan = df_ctgan.fillna(0)

print(f"\nShape finale : {df_ctgan.shape}")

# ======================
# 3. CHOISIR LA CLASSE MINORITAIRE
# ======================
# Ici 'Data Alteration' est la plus rare (204 échantillons)
minority_class = 'Data Alteration'
minority_data = df_ctgan[df_ctgan['Attack Category'] == minority_class]

print(f"\n🎯 Classe ciblée : '{minority_class}' ({len(minority_data)} échantillons)")

# ======================
# 4. TEST CTGAN (VERSION CORRIGÉE)
# ======================
if len(minority_data) >= 10:
    print(f"\n🚀 Entraînement CTGAN sur {len(minority_data)} échantillons...")
    
    # CTGAN avec paramètres réduits
    ctgan = CTGAN(
        epochs=50,
        batch_size=min(50, len(minority_data)),
        log_frequency=False,
        verbose=True
    )
    
    # !! CORRECTION ICI !!
    # On passe TOUTES les données, et on spécifie la colonne discrète
    ctgan.fit(
        train_data=minority_data,                    # Features + label ensemble
        discrete_columns=['Attack Category']         # La colonne catégorielle
    )
    
    # Génération
    n_generate = min(100, len(minority_data) * 2)
    print(f"Génération de {n_generate} échantillons...")
    
    samples = ctgan.sample(n_generate)
    
    print(f"\n✅ {len(samples)} échantillons générés !")
    print("\nAperçu des données générées :")
    print(samples.head())
    
    # Vérifier la distribution de la classe générée
    print(f"\nClasse générée : {samples['Attack Category'].unique()}")
    
    # Comparaison rapide des stats
    print(f"\nStats des données originales - {minority_class}:")
    print(minority_data[numeric_cols].describe())
    print(f"\nStats des données générées - {minority_class}:")
    print(samples[numeric_cols].describe())
    
else:
    print(f"⚠️ Pas assez d'échantillons")
