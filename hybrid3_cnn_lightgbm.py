"""
Hybride 3 — CNN + LightGBM
Deep Learning + Boosting
CNN extrait des features → LightGBM classifie
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb

NAME = "CNN + LightGBM"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

n_classes = len(le.classes_)

def build_cnn_extractor(input_dim, n_classes):
    inp = Input(shape=(input_dim, 1))
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    features = GlobalAveragePooling1D(name='features')(x)
    out = Dense(n_classes, activation='softmax')(features)
    model = Model(inputs=inp, outputs=out)
    extractor = Model(inputs=inp, outputs=features)
    return model, extractor

def run(X_tr, y_tr, X_te, label):
    X_tr_3d = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_te_3d = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)
    y_tr_cat = to_categorical(y_tr, num_classes=n_classes)

    model, extractor = build_cnn_extractor(X_tr.shape[1], n_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr_3d, y_tr_cat, epochs=10, batch_size=256, verbose=0, validation_split=0.1)

    # Extraire les features CNN
    feat_tr = extractor.predict(X_tr_3d, verbose=0)
    feat_te = extractor.predict(X_te_3d, verbose=0)

    # LightGBM sur les features CNN
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                              num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(feat_tr, y_tr)
    y_pred = clf.predict(feat_te)
    y_prob = clf.predict_proba(feat_te)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
