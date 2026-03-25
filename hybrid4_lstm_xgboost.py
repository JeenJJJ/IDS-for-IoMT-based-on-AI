"""
Hybride 4 — LSTM + XGBoost
Deep Learning + Boosting
LSTM extrait des features temporelles → XGBoost classifie
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

NAME = "LSTM + XGBoost"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

n_classes = len(le.classes_)

def run(X_tr, y_tr, X_te, label):
    X_tr_3d = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
    X_te_3d = X_te.reshape(X_te.shape[0], 1, X_te.shape[1])
    y_tr_cat = to_categorical(y_tr, num_classes=n_classes)

    inp = Input(shape=(1, X_tr.shape[1]))
    x = LSTM(64, return_sequences=False, name='lstm_features')(inp)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    extractor = Model(inputs=inp, outputs=model.get_layer('lstm_features').output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_tr_3d, y_tr_cat, epochs=10, batch_size=256, verbose=0, validation_split=0.1)

    feat_tr = extractor.predict(X_tr_3d, verbose=0)
    feat_te = extractor.predict(X_te_3d, verbose=0)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='mlogloss',
                        random_state=42, n_jobs=-1, tree_method='hist')
    xgb.fit(feat_tr, y_tr)
    y_pred = xgb.predict(feat_te)
    y_prob = xgb.predict_proba(feat_te)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
