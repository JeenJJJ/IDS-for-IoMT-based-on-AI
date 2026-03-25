"""
Hybride 2 — Isolation Forest + LSTM
Non-supervisé + Deep Learning
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

NAME = "Isolation Forest + LSTM"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

n_classes = len(le.classes_)

def run(X_tr, y_tr, X_te, label):
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso.fit(X_tr)
    score_tr = iso.decision_function(X_tr).reshape(-1, 1)
    score_te = iso.decision_function(X_te).reshape(-1, 1)

    X_tr_aug = np.hstack([X_tr, score_tr])
    X_te_aug = np.hstack([X_te, score_te])

    # Reshape pour LSTM : (samples, timesteps=1, features)
    X_tr_3d = X_tr_aug.reshape(X_tr_aug.shape[0], 1, X_tr_aug.shape[1])
    X_te_3d = X_te_aug.reshape(X_te_aug.shape[0], 1, X_te_aug.shape[1])

    y_tr_cat = to_categorical(y_tr, num_classes=n_classes)

    model = Sequential([
        LSTM(64, input_shape=(1, X_tr_aug.shape[1]), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr_3d, y_tr_cat, epochs=10, batch_size=256, verbose=0,
              validation_split=0.1)

    y_prob = model.predict(X_te_3d, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
