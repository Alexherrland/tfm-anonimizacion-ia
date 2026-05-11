"""Análisis de equidad algorítmica desagregado por subgrupo `race`."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import diffprivlib.models as dpm

from src import config
from src.preprocessing import (
    binarize_target,
    joint_ohe,
    percentile_bounds,
    percentile_data_norm,
)


def loss_disparity(accuracies_by_subgroup: pd.Series) -> float:
    """Diferencia entre el subgrupo más favorecido y el menos favorecido."""
    return float(accuracies_by_subgroup.max() - accuracies_by_subgroup.min())


def fairness_per_race_kanon(
    filepath: Path,
    k_value: int,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Desagrega la Accuracy de la Regresión Logística por subgrupo `race`."""
    if k_value == 0:
        df_anon = df_train_raw.copy()
    else:
        df_anon = pd.read_csv(filepath, sep=";")
        df_anon = df_anon[~df_anon["race"].astype(str).str.fullmatch(r"\*")].copy()

    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])

    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    scaler = StandardScaler().fit(X_anon_array)

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(scaler.transform(X_anon_array), y_anon)
    predictions = model.predict(scaler.transform(X_test_array))

    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values
    df_test_indexed = df_test_raw.reset_index(drop=True)

    rows: List[Dict] = []
    for race_value, subset in df_test_indexed.groupby("race"):
        positions = subset.index.values
        rows.append({
            "k": k_value,
            "race": race_value,
            "n": len(positions),
            "Accuracy": accuracy_score(y_test[positions], predictions[positions]),
            "F1": f1_score(y_test[positions], predictions[positions], average="weighted"),
        })
    return pd.DataFrame(rows)


def fairness_per_race_dp(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    races_test: np.ndarray,
    data_norm: float,
    epsilons: List[float] = None,
    n_repetitions: int = None,
) -> pd.DataFrame:
    """Calcula la Accuracy por raza para LR-DP a lo largo del barrido de ε."""
    epsilons = epsilons or config.EPSILON_VALUES
    n_repetitions = n_repetitions or config.N_REPETITIONS_DP

    rows: List[Dict] = []
    for epsilon in epsilons:
        for rep in range(n_repetitions):
            model = dpm.LogisticRegression(
                epsilon=epsilon,
                data_norm=data_norm,
                max_iter=100,
                random_state=config.RANDOM_STATE + rep,
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            for race_value in np.unique(races_test):
                mask = races_test == race_value
                rows.append({
                    "epsilon": epsilon,
                    "rep": rep,
                    "race": race_value,
                    "n": int(mask.sum()),
                    "Accuracy": float(accuracy_score(y_test[mask], predictions[mask])),
                })
    return pd.DataFrame(rows)
