"""Constructores de modelos baseline y diferencialmente privados."""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import diffprivlib.models as dpm

from src import config


def build_baseline_models() -> Dict[str, object]:
    """Cuatro modelos de referencia con hiperparámetros documentados en el TFM."""
    return {
        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
    }


def build_dp_models(
    epsilon: float,
    data_norm: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    random_state: int,
) -> Dict[str, object]:
    """Versiones diferencialmente privadas de los dos modelos comparables.

    max_iter=100 evita la no-convergencia de L-BFGS bajo gradientes ruidosos
    para ε altos sin penalizar el rendimiento del optimizador.
    """
    return {
        "Regresión Logística": dpm.LogisticRegression(
            epsilon=epsilon,
            data_norm=data_norm,
            max_iter=100,
            random_state=random_state,
        ),
        "Naive Bayes (Gaussian)": dpm.GaussianNB(
            epsilon=epsilon,
            bounds=bounds,
            random_state=random_state,
        ),
    }
