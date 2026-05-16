"""Fase 6 — Auditoría adversarial mediante MIA Black-Box.

Audita cinco escenarios sobre Regresión Logística: baseline,
k=10, k=50, DP con ε=0.1 y DP con ε=10.

Uso:
    python scripts/06_mia.py
"""

from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
import diffprivlib.models as dpm

from src import config
from src.data_loader import load_clean_reduced
from src.kanon import evaluate_kanon  # noqa: F401 (re-uso pipeline)
from src.mia import run_mia_blackbox
from src.plotting import plot_mia_bars
from src.preprocessing import (
    binarize_target,
    fit_scaler,
    joint_ohe,
    percentile_data_norm,
    stratified_split,
)


def _fit_lr_baseline(X_train_scaled, y_train):
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)
    return model


def _fit_lr_dp(X_train_scaled, y_train, epsilon: float, data_norm: float):
    model = dpm.LogisticRegression(
        epsilon=epsilon,
        data_norm=data_norm,
        max_iter=100,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    return model


def _build_kanon_arrays(filepath_kanon, df_test_raw):
    df_anon = pd.read_csv(filepath_kanon, sep=";")
    df_anon = df_anon[~df_anon["race"].astype(str).str.fullmatch(r"\*")]
    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])
    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_anon_array)
    return scaler.transform(X_anon_array), y_anon, scaler.transform(X_test_array)


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_KANON_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DP_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_LDIV_TCLOS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_MIA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_test_raw[column] = df_test_raw[column].astype(str)

    data_norm = percentile_data_norm(X_train_scaled, percentile=95)

    results: List[dict] = []

    # Baseline LR
    lr_baseline = _fit_lr_baseline(X_train_scaled, y_train.values)
    results.append({
        "escenario": "Baseline LR",
        **run_mia_blackbox(lr_baseline, X_train_scaled, y_train.values, X_test_scaled, y_test.values),
    })

    # k=10 y k=50
    for k_value in (10, 50):
        filepath = config.ARX_OUTPUTS_DIR / f"arx_output_k{k_value}.csv"
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite k={k_value}")
            continue
        Xa, ya, Xt = _build_kanon_arrays(filepath, df_test_raw)
        lr_k = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )
        lr_k.fit(Xa, ya)
        results.append({
            "escenario": f"LR · k={k_value}",
            **run_mia_blackbox(lr_k, Xa, ya, Xt, y_test.values),
        })

    # DP ε=0.1, ε=1 y ε=10
    for epsilon in (0.1, 1.0, 10.0):
        lr_dp = _fit_lr_dp(X_train_scaled, y_train.values, epsilon, data_norm)
        results.append({
            "escenario": f"LR · DP ε={epsilon}",
            **run_mia_blackbox(lr_dp, X_train_scaled, y_train.values, X_test_scaled, y_test.values),
        })

    df_mia = pd.DataFrame(results)
    df_mia.to_csv(config.RESULTS_MIA_DIR / "mia_results.csv", index=False)
    plot_mia_bars(df_mia, config.RESULTS_DIR / "mia_results.png")
    print(df_mia.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
