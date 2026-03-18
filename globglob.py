import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. CHARGEMENT DE TOUS LES FICHIERS
# ======================
print("="*60)
print("CHARGEMENT DU DATASET COMPLET")
print("="*60)

# Chemin vers tes dossiers (à ajuster si nécessaire)
train_path = 'data/train/'  # ou le chemin où tu as mis les fichiers

# Récupérer tous les fichiers CSV
all_files = glob.glob(os.path.join(train_path, '*.csv'))
print(f"Nombre de fichiers trouvés : {len(all_files)}")

# Charger et concaténer tous les fichiers
df_list = []
for file in tqdm(all_files, desc="Chargement des fichiers"):
    df = pd.read_csv(file)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
print(f"\nTaille totale du dataset : {len(df)} lignes")

# ======================
# 2. ANALYSE DE LA DISTRIBUTION
# ======================
print("\n" + "="*60)
print("ANALYSE DE LA DISTRIBUTION DES CLASSES")
print("="*60)

# Vérifier le nom de la colonne label (parfois 'Label', 'label', 'Class')
label_col = None
for col in df.columns:
    if col.lower() in ['label', 'class', 'attack_type']:
        label_col = col
        break

if label_col is None:
    raise ValueError("Colonne de label non trouvée. Vérifie les noms de colonnes.")

print(f"Colonne label identifiée : '{label_col}'")

# Distribution des classes
class_dist = df[label_col].value_counts()
print("\nDistribution initiale :")
for cls, count in class_dist.items():
    percentage = 100 * count / len(df)
    print(f"  {cls}: {count} ({percentage:.2f}%)")

# Identifier la classe minoritaire (celle avec le moins d'échantillons)
minority_class = class_dist.index[-1]
minority_count = class_dist.iloc[-1]
print(f"\n🎯 Classe minoritaire : '{minority_class}' avec {minority_count} échantillons")

# ======================
# 3. PRÉPARATION DES DONNÉES POUR TINYGAN
# ======================
print("\n" + "="*60)
print("PRÉPARATION DES DONNÉES POUR TINYGAN")
print("="*60)

# Séparer features (X) et labels (y)
X = df.drop(columns=[label_col])
y = df[label_col]

# Encoder les labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardiser les features (important pour les GANs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Extraire les données de la classe minoritaire
minority_mask = (y == minority_class)
X_minority = X_scaled[minority_mask]

print(f"Données de la classe minoritaire : {X_minority.shape}")

# Convertir en tenseurs PyTorch
X_minority_tensor = torch.FloatTensor(X_minority)

# Créer un DataLoader pour l'entraînement
batch_size = min(64, len(X_minority))
minority_dataset = TensorDataset(X_minority_tensor)
minority_loader = DataLoader(minority_dataset, batch_size=batch_size, shuffle=True)

# ======================
# 4. DÉFINITION DE TINYGAN
# ======================
print("\n" + "="*60)
print("DÉFINITION DE L'ARCHITECTURE TINYGAN")
print("="*60)

class TinyGenerator(nn.Module):
    """Générateur ultra-léger - coeur de TinyGAN"""
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Pour éviter le mode collapse
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class TinyDiscriminator(nn.Module):
    """Discriminateur léger"""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Paramètres
noise_dim = 64  # Dimension du bruit d'entrée
data_dim = X_minority.shape[1]  # Nombre de features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Dimension des données : {data_dim}")
print(f"Dimension du bruit : {noise_dim}")
print(f"Utilisation de : {device}")

# Initialisation
G = TinyGenerator(noise_dim, data_dim).to(device)
D = TinyDiscriminator(data_dim).to(device)

# Optimiseurs
G_optim = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Perte
criterion = nn.BCELoss()

# ======================
# 5. ENTRAÎNEMENT
# ======================
print("\n" + "="*60)
print("ENTRAÎNEMENT DE TINYGAN")
print("="*60)

epochs = 1000  # À ajuster selon la convergence
n_critic = 5   # Entraîner le discriminateur plus souvent

# Pour suivre la progression
G_losses = []
D_losses = []

for epoch in tqdm(range(epochs), desc="Entraînement"):
    for batch_idx, (real_data,) in enumerate(minority_loader):
        real_data = real_data.to(device)
        batch_size_current = real_data.size(0)
        
        # Labels pour la perte
        real_labels = torch.ones(batch_size_current, 1).to(device)
        fake_labels = torch.zeros(batch_size_current, 1).to(device)
        
        # --- ENTRAÎNEMENT DU DISCRIMINATEUR ---
        for _ in range(n_critic):
            # Données réelles
            real_pred = D(real_data)
            real_loss = criterion(real_pred, real_labels)
            
            # Données fausses
            z = torch.randn(batch_size_current, noise_dim).to(device)
            fake_data = G(z)
            fake_pred = D(fake_data.detach())
            fake_loss = criterion(fake_pred, fake_labels)
            
            # Perte totale du discriminateur
            D_loss = real_loss + fake_loss
            
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()
        
        # --- ENTRAÎNEMENT DU GÉNÉRATEUR ---
        z = torch.randn(batch_size_current, noise_dim).to(device)
        fake_data = G(z)
        fake_pred = D(fake_data)
        G_loss = criterion(fake_pred, real_labels)  # Veut tromper D
        
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
    
    # Sauvegarde des pertes
    G_losses.append(G_loss.item())
    D_losses.append(D_loss.item())
    
    # Affichage périodique
    if (epoch + 1) % 100 == 0:
        print(f"\nEpoch [{epoch+1}/{epochs}] - D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")

# ======================
# 6. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ======================
print("\n" + "="*60)
print("GÉNÉRATION DE DONNÉES SYNTHÉTIQUES")
print("="*60)

# Combien d'échantillons générer pour équilibrer ?
# On veut que la classe minoritaire ait au moins le même nombre que la classe majoritaire
majority_count = class_dist.iloc[0]
samples_to_generate = majority_count - minority_count

print(f"Classe majoritaire : {class_dist.index[0]} avec {majority_count} échantillons")
print(f"Échantillons à générer pour '{minority_class}' : {samples_to_generate}")

# Générer en plusieurs batches pour éviter les problèmes mémoire
G.eval()
generated_batches = []
batch_size_gen = 1000

with torch.no_grad():
    for i in range(0, samples_to_generate, batch_size_gen):
        current_batch = min(batch_size_gen, samples_to_generate - i)
        z = torch.randn(current_batch, noise_dim).to(device)
        fake_batch = G(z).cpu().numpy()
        generated_batches.append(fake_batch)

generated_data_scaled = np.vstack(generated_batches)

# Dé-normaliser
generated_data = scaler.inverse_transform(generated_data_scaled)

# Créer un DataFrame avec les données générées
generated_df = pd.DataFrame(generated_data, columns=X.columns)
generated_df[label_col] = minority_class

print(f"✅ {len(generated_df)} échantillons synthétiques générés pour la classe '{minority_class}'")

# ======================
# 7. CRÉATION DU DATASET ÉQUILIBRÉ
# ======================
print("\n" + "="*60)
print("CRÉATION DU DATASET ÉQUILIBRÉ")
print("="*60)

# Option 1: Ajouter les données générées aux originales
df_balanced = pd.concat([df, generated_df], ignore_index=True)

# Option 2: Si tu veux remplacer la classe minoritaire par les données générées
# df_balanced = pd.concat([df[df[label_col] != minority_class], generated_df])

print("\nNouvelle distribution :")
new_dist = df_balanced[label_col].value_counts()
for cls, count in new_dist.items():
    percentage = 100 * count / len(df_balanced)
    print(f"  {cls}: {count} ({percentage:.2f}%)")

# Sauvegarder
output_file = 'iomt_dataset_balanced_tinygan.csv'
df_balanced.to_csv(output_file, index=False)
print(f"\n✅ Dataset équilibré sauvegardé dans '{output_file}'")

# ======================
# 8. VISUALISATION RAPIDE (optionnelle)
# ======================
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(class_dist.index[:5], class_dist.values[:5])
    plt.title('Distribution originale (top 5)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(new_dist.index[:5], new_dist.values[:5])
    plt.title('Distribution après TinyGAN (top 5)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.show()
    print("✅ Graphique sauvegardé dans 'distribution_comparison.png'")
except:
    print("📊 Matplotlib non disponible, passage à l'étape suivante")

print("\n" + "="*60)
print("🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS")
print("="*60)
