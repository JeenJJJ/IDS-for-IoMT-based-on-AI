import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
from ctgan import CTGAN
import matplotlib.pyplot as plt
import os

TRAIN_PATH        = "train_iomt.csv"
TEST_PATH         = "test_iomt.csv"
LABEL_COL         = "label"
SAMPLES_PER_CLASS = 2000
CTGAN_EPOCHS      = 100


def load_and_prepare():
    print("Chargement dataset...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    def clean(data):
        X = data.drop(columns=[LABEL_COL])
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        return X

    # Sample ou augmente chaque classe a exactement SAMPLES_PER_CLASS
    frames = []
    for cls, group in df_train.groupby(LABEL_COL):
        if len(group) >= SAMPLES_PER_CLASS:
            # Assez de data : on sample
            frames.append(group.sample(SAMPLES_PER_CLASS, random_state=42))
        else:
            # Pas assez : on garde tout + on complete avec CTGAN
            n_missing = SAMPLES_PER_CLASS - len(group)
            print(f"  CTGAN '{cls}' : {len(group)} samples -> genere {n_missing}...")
            subset = clean(group)
            subset_with_label = subset.copy()
            subset_with_label[LABEL_COL] = cls
            ctgan = CTGAN(epochs=CTGAN_EPOCHS, batch_size=500, verbose=False)
            ctgan.fit(subset_with_label, discrete_columns=[LABEL_COL])
            synthetic = ctgan.sample(n_missing)
            del ctgan
            frames.append(pd.concat([group, synthetic], ignore_index=True))

    df_train = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

    print(f"\nTrain: {df_train.shape} | Test: {df_test.shape}")
    print("\nDistribution train finale:")
    print(df_train[LABEL_COL].value_counts())

    X_train_raw = clean(df_train)
    X_test_raw  = clean(df_test)

    le = LabelEncoder()
    le.fit(pd.concat([df_train[LABEL_COL], df_test[LABEL_COL]]))
    y_train = le.transform(df_train[LABEL_COL])
    y_test  = le.transform(df_test[LABEL_COL])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            le, X_train_raw.shape[1])


def evaluate_and_plot(name, y_test, y_pred, y_prob, le, output_dir="."):
    print(f"\n{'='*50}")
    print(f"RESULTATS — {name}")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    n_classes = len(le.classes_)
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}", linewidth=2)
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=range(n_classes))
        for i, cls in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Taux Faux Positifs")
    ax.set_ylabel("Taux Vrais Positifs")
    ax.set_title(f"Courbe ROC — {name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fname = os.path.join(output_dir, f"roc_{name.replace(' ', '_').replace('+', '')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC sauvegardee: {fname}")


