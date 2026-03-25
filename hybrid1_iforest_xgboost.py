"""
Hybride 1 — Isolation Forest + XGBoost
Non-supervisé + Boosting
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

NAME = "Isolation Forest + XGBoost"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

def run(X_tr, y_tr, X_te, label):
    # Isolation Forest : génère un score d'anomalie comme feature supplémentaire
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso.fit(X_tr)
    score_tr = iso.decision_function(X_tr).reshape(-1, 1)
    score_te = iso.decision_function(X_te).reshape(-1, 1)

    X_tr_aug = np.hstack([X_tr, score_tr])
    X_te_aug = np.hstack([X_te, score_te])

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='mlogloss',
                        random_state=42, n_jobs=-1, tree_method='hist')
    xgb.fit(X_tr_aug, y_tr)
    y_pred = xgb.predict(X_te_aug)
    y_prob = xgb.predict_proba(X_te_aug)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
