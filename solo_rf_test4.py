import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ctgan import CTGAN

# ======================
# CONFIGURATION
# ======================
DATA_PATH    = "wustl-ehms-2020_with_attacks_categories.csv"
LABEL_COL    = "Attack Category"
TARGET       = 10_000
MINORITY_THR = 10_000
CTGAN_EPOCHS = 100
CHUNK        = 500_000

# ======================
# CHARGEMENT DU DATASET
# ======================
print("Chargement du dataset...")
frames = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK):
    frames.append(chunk)
df = pd.concat(frames, ignore_index=True)
print(f"Dataset chargé : {df.shape}")
print("\nDistribution originale :")
print(df[LABEL_COL].value_counts())

# ======================
# NETTOYAGE
# ======================
def clean(data, ref_cols=None):
    X = data.drop(columns=[LABEL_COL])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    return X

X = clean(df)
le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])

# ======================
# SPLIT TRAIN / TEST STRATIFIÉ
# ======================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {X_train_raw.shape} | Test : {X_test_raw.shape}")

# ======================
# CTGAN EQUILIBRAGE (classes minoritaires seulement)
# ======================
print("\nEntraînement CTGAN sur classes minoritaires...")
df_train = X_train_raw.copy()
df_train[LABEL_COL] = le.inverse_transform(y_train)

dist = df_train[LABEL_COL].value_counts()
synth_frames = []

for cls, count in dist.items():
    if count < MINORITY_THR:
        subset = df_train[df_train[LABEL_COL] == cls]
        if len(subset) < 10:
            print(f"  ⚠️  {cls} : {len(subset)} samples, ignoré")
            continue
        n = TARGET - count
        print(f"  → '{cls}' ({count} samples, génère {n})...")
        ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=500, verbose=False)
        ctgan.fit(subset, discrete_columns=[LABEL_COL])
        synth_frames.append(ctgan.sample(n))
        del ctgan

df_balanced = pd.concat([df_train] + synth_frames, ignore_index=True)
print("\nDistribution après CTGAN :")
print(df_balanced[LABEL_COL].value_counts())

X_bal = clean(df_balanced, ref_cols=X_train_raw.columns)
y_bal = le.transform(df_balanced[LABEL_COL])

# ======================
# RANDOM FOREST UNIQUEMENT
# ======================
def rf_classifier(X_tr, y_tr, X_te):
    print("  Entraînement Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)

# ======================
# SANS CTGAN
# ======================
print("\n" + "="*50)
print("SANS CTGAN — Random Forest")
print("="*50)
y_pred_no_ctgan = rf_classifier(X_train_raw, y_train, X_test_raw)
print(classification_report(y_test, y_pred_no_ctgan, target_names=le.classes_))

# ======================
# AVEC CTGAN
# ======================
print("\n" + "="*50)
print("AVEC CTGAN — Random Forest")
print("="*50)
y_pred_ctgan = rf_classifier(X_bal, y_bal, X_test_raw)
print(classification_report(y_test, y_pred_ctgan, target_names=le.classes_))
