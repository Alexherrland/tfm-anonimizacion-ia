"""Fase 7 — Tests de significancia estadística.

Calcula los tests de McNemar para los modelos k-anonimizados y los
tests de Wilcoxon de una muestra para los modelos diferencialmente
privados.

Uso:
    python scripts/07_statistical_tests.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import load_clean_reduced
from src.kanon import evaluate_kanon  # noqa
from src.models import build_baseline_models
from src.preprocessing import (
    binarize_target,
    fit_scaler,
    joint_ohe,
    load_baselines,
    stratified_split,
)
from src.statistical_tests import build_mcnemar_table, build_wilcoxon_table


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

    # Predicciones del baseline para cada modelo
    baseline_predictions = {}
    for name, model in build_baseline_models().items():
        model.fit(X_train_scaled, y_train)
        baseline_predictions[name] = model.predict(X_test_scaled)

    # Predicciones por modelo y nivel de k
    anonymized_predictions = {}
    for k_value in config.K_VALUES:
        filepath = config.ARX_OUTPUTS_DIR / f"arx_output_k{k_value}.csv"
        if not filepath.exists():
            continue
        df_anon = pd.read_csv(filepath, sep=";")
        df_anon = df_anon[~df_anon["race"].astype(str).str.fullmatch(r"\*")]
        y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
        X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
        X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])
        Xa_array, Xt_array = joint_ohe(X_anon, X_test)
        sc = StandardScaler().fit(Xa_array)
        for name, model in build_baseline_models().items():
            model.fit(sc.transform(Xa_array), y_anon)
            anonymized_predictions[(name, k_value)] = model.predict(sc.transform(Xt_array))

    df_mcnemar = build_mcnemar_table(
        baseline_predictions=baseline_predictions,
        anonymized_predictions=anonymized_predictions,
        y_test=y_test.values,
    )
    df_mcnemar.to_csv(config.RESULTS_KANON_DIR / "mcnemar_kanon.csv", index=False)
    print("McNemar (baseline vs k-anon):")
    print(df_mcnemar.round(4).to_string(index=False))

    # Test de Wilcoxon sobre los resultados DP previamente guardados
    dp_path = config.RESULTS_DP_DIR / "resultados_dp.csv"
    if dp_path.exists():
        df_dp = pd.read_csv(dp_path)
        # Baselines leídos de resultados_kanon.csv (k=0) — fuente única de verdad
        baselines_full = load_baselines()
        baselines = {model: acc for model, (acc, _) in baselines_full.items()}
        df_wilcoxon = build_wilcoxon_table(df_dp, baselines)
        df_wilcoxon.to_csv(config.RESULTS_DP_DIR / "wilcoxon_dp.csv", index=False)
        print("\nWilcoxon (DP vs baseline):")
        print(df_wilcoxon.round(4).to_string(index=False))
    else:
        print(f"\nAviso: no se encuentra {dp_path}, se omiten los tests Wilcoxon")


if __name__ == "__main__":
    main()
