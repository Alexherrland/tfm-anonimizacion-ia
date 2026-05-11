"""Ingesta y limpieza del conjunto Diabetes 130-US hospitals."""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src import config


def load_raw_dataset() -> pd.DataFrame:
    """Descarga el dataset original desde el repositorio UCI."""
    dataset = fetch_ucirepo(id=config.DATASET_ID)
    return pd.concat([dataset.data.features, dataset.data.targets], axis=1)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las tres estrategias de preprocesamiento descritas en la memoria.

    1. Suprime la columna `weight` (96,85 % de nulos).
    2. Imputa con la categoría 'No_Registrado' los nulos clínicos semánticos.
    3. Elimina filas con nulos residuales preservando la variable `race`.
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.drop(columns=["weight"], inplace=True)
    df[config.IMPUTE_COLUMNS] = df[config.IMPUTE_COLUMNS].fillna(config.IMPUTE_VALUE)
    df.dropna(subset=config.DROP_NA_SUBSET, inplace=True)
    return df


def map_diagnosis(code) -> str:
    """Mapea un código CIE-9 a una de las nueve macrocategorías clínicas."""
    if pd.isna(code) or code == "?":
        return "Other"
    if isinstance(code, str) and (code.startswith("V") or code.startswith("E")):
        return "Other"
    try:
        value = float(code)
    except (TypeError, ValueError):
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if (390 <= value <= 459) or value == 785:
        return "Circulatory"
    if (460 <= value <= 519) or value == 786:
        return "Respiratory"
    if (520 <= value <= 579) or value == 787:
        return "Digestive"
    if (580 <= value <= 629) or value == 788:
        return "Genitourinary"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 800 <= value <= 999:
        return "Injury"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def reduce_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce los códigos CIE-9 originales a 9 macrocategorías médicas.

    Esta agrupación semántica reduce el espacio One-Hot Encoding
    de 2.405 a 121 dimensiones, condición previa para la aplicación
    de Privacidad Diferencial sin colapso de utilidad.
    """
    df = df.copy()
    for column in ("diag_1", "diag_2", "diag_3"):
        df[f"{column}_category"] = df[column].apply(map_diagnosis)
    return df.drop(columns=config.DROP_COLUMNS_FOR_ANONYMIZATION)


def load_clean_reduced() -> pd.DataFrame:
    """Pipeline completo: descarga, limpieza y agrupación de diagnósticos."""
    df = load_raw_dataset()
    df = clean_dataset(df)
    df = reduce_diagnoses(df)
    return df
