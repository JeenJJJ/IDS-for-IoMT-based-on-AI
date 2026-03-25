"""
Hybride 5 — AdaBoost + CatBoost
Boosting + Boosting
AdaBoost génère des features de probabilité → CatBoost classifie
"""
import numpy as np
from pipeline_base import load_and_prepare, evaluate_and_plot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

NAME = "AdaBoost + CatBoost"

(X_train, X_test, y_train, y_test,
 X_bal, y_bal, le, n_features) = load_and_prepare()

def run(X_tr, y_tr, X_te, label):
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=50, random_state=42
    )
    ada.fit(X_tr, y_tr)
    prob_tr = ada.predict_proba(X_tr)
    prob_te = ada.predict_proba(X_te)

    # Concat features originales + probas AdaBoost
    X_tr_aug = np.hstack([X_tr, prob_tr])
    X_te_aug = np.hstack([X_te, prob_te])

    cat = CatBoostClassifier(iterations=200, learning_rate=0.05,
                             depth=6, random_seed=42, verbose=0)
    cat.fit(X_tr_aug, y_tr)
    y_pred = cat.predict(X_te_aug).flatten()
    y_prob = cat.predict_proba(X_te_aug)
    evaluate_and_plot(f"{NAME} ({label})", y_test, y_pred, y_prob, le, output_dir="results")

print("\n--- SANS CTGAN ---")
run(X_train, y_train, X_test, "sans CTGAN")

print("\n--- AVEC CTGAN ---")
run(X_bal, y_bal, X_test, "avec CTGAN")
