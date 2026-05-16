"""Auditoría adversarial mediante ataques de inferencia de membresía (MIA)."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from src import config


def run_mia_blackbox(
    sklearn_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_train_sample: int = 10000,
    n_test_sample: int = 5000,
    seed: int = None,
) -> Dict[str, float]:
    """Ejecuta un MIA Black-Box sobre un modelo de Regresión Logística entrenado.

    Sigue la metodología de Shokri et al. (2017): el atacante construye un
    clasificador de pertenencia (Random Forest) sobre la mitad del conjunto
    auditor y se evalúa sobre la mitad restante.
    """
    seed = seed if seed is not None else config.RANDOM_STATE
    rng = np.random.default_rng(seed)

    n_tr = min(len(X_train), n_train_sample)
    n_te = min(len(X_test), n_test_sample)
    idx_tr = rng.choice(len(X_train), n_tr, replace=False)
    idx_te = rng.choice(len(X_test), n_te, replace=False)

    X_tr_sample = X_train[idx_tr]
    y_tr_sample = y_train[idx_tr]
    X_te_sample = X_test[idx_te]
    y_te_sample = y_test[idx_te]

    art_classifier = ScikitlearnLogisticRegression(model=sklearn_model)
    attack = MembershipInferenceBlackBox(estimator=art_classifier, attack_model_type="rf")

    half_tr = n_tr // 2
    half_te = n_te // 2

    attack.fit(
        x=X_tr_sample[:half_tr],
        y=y_tr_sample[:half_tr],
        test_x=X_te_sample[:half_te],
        test_y=y_te_sample[:half_te],
    )

    inferred_members = attack.infer(X_tr_sample[half_tr:], y_tr_sample[half_tr:])
    inferred_non_members = attack.infer(X_te_sample[half_te:], y_te_sample[half_te:])

    # Probabilidades del clasificador atacante para calcular AUC
    proba_members = attack.infer(X_tr_sample[half_tr:], y_tr_sample[half_tr:], probabilities=True)
    proba_non_members = attack.infer(X_te_sample[half_te:], y_te_sample[half_te:], probabilities=True)
    # ART devuelve probabilidad de la clase "member"; según versión puede ser (n,) o (n,1)/(n,2)
    proba_members = np.asarray(proba_members).reshape(len(proba_members), -1)
    proba_non_members = np.asarray(proba_non_members).reshape(len(proba_non_members), -1)
    # Tomar la última columna (prob. de "member" según convención ART)
    scores = np.concatenate([proba_members[:, -1], proba_non_members[:, -1]])
    labels = np.concatenate([
        np.ones(len(proba_members), dtype=int),
        np.zeros(len(proba_non_members), dtype=int),
    ])
    try:
        attack_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        attack_auc = float("nan")

    n_correct = int((inferred_members == 1).sum()) + int((inferred_non_members == 0).sum())
    n_total = len(inferred_members) + len(inferred_non_members)
    attack_accuracy = n_correct / n_total

    rate_member = float((inferred_members == 1).mean())
    rate_non_member = float((inferred_non_members == 1).mean())
    advantage = rate_member - rate_non_member

    utility = float(accuracy_score(y_test, sklearn_model.predict(X_test)))

    return {
        "utility_acc": round(utility, 4),
        "attack_acc": round(attack_accuracy, 4),
        "attack_advantage": round(advantage, 4),
        "attack_auc": round(attack_auc, 4),
        "rate_member_predicted_1": round(rate_member, 4),
        "rate_non_member_predicted_1": round(rate_non_member, 4),
    }
