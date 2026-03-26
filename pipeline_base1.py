import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             RocCurveDisplay, roc_curve, auc)
from ctgan import CTGAN

DATA_PATH    = "wustl-ehms-2020_with_attacks_categories.csv"
LABEL_COL    = "Attack Category"
TARGET       = 11_000
MINORITY_THR = 10_000
CTGAN_EPOCHS = 500
CHUNK        = 500_000


def load_and_prepare():
    print("Chargement dataset...")
    frames = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK):
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    print(f"Dataset chargé : {df.shape}")
    print("\nDistribution originale :")
    print(df[LABEL_COL].value_counts())

    print("\nNettoyage")
    def clean(data, ref_cols=None):
        X = data.drop(columns=[LABEL_COL, 'Label'], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        if ref_cols is not None:
            X = X.reindex(columns=ref_cols, fill_value=0)
        return X

    X = clean(df)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {X_train_raw.shape} | Test : {X_test_raw.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # CTGAN
    print("CTGAN équilibrage...")
    df_train = X_train_raw.copy()
    df_train[LABEL_COL] = le.inverse_transform(y_train)
    dist = df_train[LABEL_COL].value_counts()
    synth_frames = []
    for cls, count in dist.items():
        if count < MINORITY_THR:
            subset = df_train[df_train[LABEL_COL] == cls]
            if len(subset) < 10:
                continue
            n = TARGET - count
            print(f"  → '{cls}' ({count} samples, génère {n})...")
            ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=200, verbose=False)
            ctgan.fit(subset, discrete_columns=[LABEL_COL])
            synth_frames.append(ctgan.sample(n))
            del ctgan

    df_balanced = pd.concat([df_train] + synth_frames, ignore_index=True)
    print("\nDistribution après CTGAN :")
    print(df_balanced[LABEL_COL].value_counts())

    X_bal = clean(df_balanced, ref_cols=X_train_raw.columns)
    y_bal = le.transform(df_balanced[LABEL_COL])
    X_bal_scaled = scaler.transform(X_bal)

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            X_bal_scaled, y_bal, le, X_train_raw.shape[1])


def evaluate_and_plot(name, y_test, y_pred, y_prob, le, output_dir="."):
    """Affiche les KPI et sauvegarde la courbe ROC."""
    print(f"\n{'='*50}")
    print(f"RÉSULTATS — {name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    n_classes = len(le.classes_)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

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
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fname = os.path.join(output_dir, f"roc_{name.replace(' ', '_').replace('+', '')}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbe ROC sauvegardée : {fname}")


