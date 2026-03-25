"""
Hybride 6 — Random Forest + SVM
ML + ML
RF génère des features (feuilles) → SVM classifie
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

NAME = "Random Forest + SVM"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

def run(X_tr, y_tr, X_te, label):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    # Utiliser les probas RF comme features pour SVM
    prob_tr = rf.predict_proba(X_tr)
    prob_te = rf.predict_proba(X_te)

    # Re-scaler pour SVM
    sc2 = StandardScaler()
    prob_tr_s = sc2.fit_transform(prob_tr)
    prob_te_s  = sc2.transform(prob_te)

    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              probability=True, random_state=42)
    svm.fit(prob_tr_s, y_tr)
    y_pred = svm.predict(prob_te_s)
    y_prob = svm.predict_proba(prob_te_s)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
