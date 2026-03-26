"""
pipeline_base.py — Pipeline commun pour tous les hybrides
Dataset : CICIoMT2024 (train_iomt.csv / test_iomt.csv)

Stratégie mémoire :
  - Chargement chunk par chunk avec reservoir sampling
  - Train : 50 000 lignes aléatoires
  - Test  : 20 000 lignes aléatoires
  - CTGAN : augmentation jusqu'à 5 000 échantillons
             uniquement pour les classes ayant > 1 000 exemples
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
from ctgan import CTGAN

# ── Chemins & constantes ─────────────────────────────────────────────────────
TRAIN_PATH      = "train_iomt.csv"
TEST_PATH       = "test_iomt.csv"
LABEL_COL       = "label"

TRAIN_SAMPLE    = 50_000    # lignes à garder du train
TEST_SAMPLE     = 20_000    # lignes à garder du test

CTGAN_TARGET    = 5_000     # taille cible après augmentation
CTGAN_MIN_COUNT = 1_000     # seuil min pour déclencher CTGAN
CTGAN_EPOCHS    = 100
CHUNK           = 200_000   # taille des chunks (adapté ~10 Go RAM)


# ── Reservoir sampling (niveau chunk, efficace en mémoire) ───────────────────
def reservoir_sample_csv_fast(path, n_samples, chunk_size=CHUNK, seed=42):
    """
    Lit le CSV par chunks et applique un reservoir sampling approché
    pour obtenir ~n_samples lignes sans tout charger en mémoire.
    """
    np.random.seed(seed)
    reservoir = []
    total_seen = 0

    print(f"  Lecture de '{path}' par chunks de {chunk_size:,}...")

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk = chunk.reset_index(drop=True)
        n_chunk = len(chunk)
        total_seen += n_chunk

        # Probabilité d'accepter chaque ligne de ce chunk
        keep_prob = min(1.0, n_samples / total_seen)
        mask = np.random.rand(n_chunk) < keep_prob
        kept = chunk[mask]

        if len(kept) > 0:
            reservoir.append(kept)

        # Fusion et sous-échantillonnage si on dépasse la cible
        if sum(len(r) for r in reservoir) > n_samples * 1.5:
            combined = pd.concat(reservoir, ignore_index=True)
            if len(combined) > n_samples:
                combined = combined.sample(n=n_samples, random_state=seed).reset_index(drop=True)
            reservoir = [combined]

    if not reservoir:
        raise ValueError(f"Aucune donnée lue depuis '{path}'")

    result = pd.concat(reservoir, ignore_index=True)

    # Ajustement final
    if len(result) > n_samples:
        result = result.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"  → {total_seen:,} lignes vues, {len(result):,} retenues.")
    return result


# ── Nettoyage features ───────────────────────────────────────────────────────
def clean_features(df, label_col, ref_cols=None):
    """
    Supprime le label, garde uniquement les colonnes numériques,
    remplace inf/nan par la médiane.
    Si ref_cols fourni, réindexe pour assurer la cohérence train/test.
    """
    X = df.drop(columns=[label_col], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    return X


# ── Fonction principale ──────────────────────────────────────────────────────
def load_and_prepare():
    """
    Charge et prépare les données CICIoMT2024 :
      1. Reservoir sampling sur train (50k) et test (20k)
      2. Encodage du label
      3. StandardScaler (fit sur train uniquement)
      4. CTGAN sur les classes éligibles (> 1 000 exemples, < 5 000)

    Retourne :
      X_train, X_test, y_train, y_test,
      X_bal, y_bal, le, n_features
    """

    # ── 1. Chargement ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNÉES — CICIoMT2024")
    print("="*60)

    print(f"\n[1/4] Train set (cible : {TRAIN_SAMPLE:,} lignes)...")
    df_train = reservoir_sample_csv_fast(TRAIN_PATH, TRAIN_SAMPLE, seed=42)

    print(f"\n[2/4] Test set (cible : {TEST_SAMPLE:,} lignes)...")
    df_test = reservoir_sample_csv_fast(TEST_PATH, TEST_SAMPLE, seed=42)

    # ── 2. Encodage du label ──────────────────────────────────────────────────
    print("\n[3/4] Encodage des labels...")

    df_train[LABEL_COL] = df_train[LABEL_COL].astype(str).str.strip()
    df_test[LABEL_COL]  = df_test[LABEL_COL].astype(str).str.strip()

    le = LabelEncoder()
    # Fit sur l'union train + test pour cohérence des indices de classes
    all_labels = pd.concat([df_train[LABEL_COL], df_test[LABEL_COL]], ignore_index=True)
    le.fit(all_labels)

    y_train = le.transform(df_train[LABEL_COL])
    y_test  = le.transform(df_test[LABEL_COL])

    print(f"  {len(le.classes_)} classes : {list(le.classes_)}")

    dist = df_train[LABEL_COL].value_counts()
    print("\n  Distribution train échantillonné :")
    for cls, cnt in dist.items():
        marker = "✓ CTGAN" if CTGAN_MIN_COUNT < cnt < CTGAN_TARGET else (
                 "— déjà OK" if cnt >= CTGAN_TARGET else "✗ ignoré")
        print(f"    {cls:<30} : {cnt:>6,}  [{marker}]")

    # ── 3. Features & scaling ─────────────────────────────────────────────────
    X_train_raw = clean_features(df_train, LABEL_COL)
    ref_cols    = X_train_raw.columns.tolist()
    X_test_raw  = clean_features(df_test, LABEL_COL, ref_cols=ref_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    n_features = X_train_raw.shape[1]
    print(f"\n  Features : {n_features}  |  Train : {X_train_scaled.shape}  |  Test : {X_test_scaled.shape}")

    # ── 4. CTGAN ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] CTGAN — seuil éligibilité : ]{CTGAN_MIN_COUNT:,} ; {CTGAN_TARGET:,}[")

    df_train_copy = X_train_raw.copy()
    df_train_copy[LABEL_COL] = df_train[LABEL_COL].values

    synth_frames = []

    for cls, count in dist.items():
        if count <= CTGAN_MIN_COUNT:
            print(f"  ✗ '{cls}' ({count:,}) — sous le seuil, ignoré")
            continue
        if count >= CTGAN_TARGET:
            print(f"  — '{cls}' ({count:,}) — déjà suffisant")
            continue

        n_to_gen = CTGAN_TARGET - count
        print(f"  ↑ '{cls}' ({count:,}) — génération de {n_to_gen:,} exemples...")

        subset = df_train_copy[df_train_copy[LABEL_COL] == cls].copy()
        try:
            ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=500, verbose=False)
            ctgan.fit(subset, discrete_columns=[LABEL_COL])
            synth = ctgan.sample(n_to_gen)
            synth_frames.append(synth)
            del ctgan
        except Exception as e:
            print(f"    ⚠ Erreur CTGAN pour '{cls}' : {e}")

    if synth_frames:
        df_balanced = pd.concat([df_train_copy] + synth_frames, ignore_index=True)
        print(f"\n  Taille dataset équilibré : {len(df_balanced):,} lignes")
    else:
        df_balanced = df_train_copy.copy()
        print("\n  Aucune augmentation effectuée.")

    X_bal_raw    = clean_features(df_balanced, LABEL_COL, ref_cols=ref_cols)
    y_bal        = le.transform(df_balanced[LABEL_COL].astype(str).str.strip())
    X_bal_scaled = scaler.transform(X_bal_raw)

    print("\n" + "="*60)
    print(f"  X_train : {X_train_scaled.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test_scaled.shape}   |  y_test  : {y_test.shape}")
    print(f"  X_bal   : {X_bal_scaled.shape}    |  y_bal   : {y_bal.shape}")
    print("="*60 + "\n")

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            X_bal_scaled, y_bal, le, n_features)


# ── Évaluation & ROC ─────────────────────────────────────────────────────────
def evaluate_and_plot(name, y_test, y_pred, y_prob, le, output_dir="."):
    """Affiche les métriques et sauvegarde la courbe ROC."""
    print(f"\n{'='*60}")
    print(f"RÉSULTATS — {name}")
    print('='*60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    n_classes = len(le.classes_)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=range(n_classes))
        for i, cls in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_score:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel("Taux Faux Positifs")
    ax.set_ylabel("Taux Vrais Positifs")
    ax.set_title(f"Courbe ROC — {name}")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    safe_name = name.replace(' ', '_').replace('+', '').replace('/', '_')
    fname = os.path.join(output_dir, f"roc_{safe_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe ROC sauvegardée : {fname}")
