"""División train/test, codificación One-Hot y estandarización."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


def binarize_target(series: pd.Series) -> pd.Series:
    """Convierte la variable readmitted en clasificación binaria 0/1."""
    return (series != "NO").astype(int)


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición estratificada con la semilla fijada en config.

    Devuelve (X_train_encoded, X_test_encoded, y_train, y_test) donde
    las dos primeras son matrices OHE aplicadas sobre el conjunto completo
    antes del split para garantizar columnas alineadas.
    """
    X_raw = df.drop(columns=[config.TARGET_COLUMN])
    y_bin = binarize_target(df[config.TARGET_COLUMN])
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    return train_test_split(
        X_encoded,
        y_bin,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_bin,
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajusta un StandardScaler sobre el conjunto de entrenamiento."""
    return StandardScaler().fit(X_train)


def joint_ohe(X_anon: pd.DataFrame, X_test_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica One-Hot Encoding conjunto sobre la unión train_anon ∪ test.

    Garantiza un column space idéntico entre el conjunto anonimizado
    y el conjunto de prueba en claro. Necesario para evaluar modelos
    entrenados sobre datos generalizados frente a datos originales.
    """
    n_anon = len(X_anon)
    stacked = pd.concat([X_anon, X_test_raw], ignore_index=True)
    encoded = pd.get_dummies(stacked, drop_first=True)
    return encoded.iloc[:n_anon].values, encoded.iloc[n_anon:].values


def percentile_data_norm(X_scaled: np.ndarray, percentile: int = 95) -> float:
    """Devuelve el percentil indicado de las normas L2 de las filas.

    Esta cota recortada se utiliza como `data_norm` en `diffprivlib`
    para evitar la sobrecalibración del ruido por filas atípicas.
    """
    norms = np.linalg.norm(X_scaled, axis=1)
    return float(np.percentile(norms, percentile))


def percentile_bounds(X_scaled: np.ndarray, low: int = 1, high: int = 99) -> Tuple[np.ndarray, np.ndarray]:
    """Cotas robustas por característica para diffprivlib.GaussianNB."""
    return np.percentile(X_scaled, low, axis=0), np.percentile(X_scaled, high, axis=0)
