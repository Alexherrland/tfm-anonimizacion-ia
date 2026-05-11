"""Fase 3 — Evaluación post-ARX y análisis de equidad.

Lee los CSVs anonimizados arx_output_k<k>.csv producidos manualmente
en ARX Desktop, reentrena los modelos sobre cada versión y produce
las tablas y figuras del Capítulo 4.

Uso:
    python scripts/03_evaluate_kanon.py
"""

import pandas as pd

from src import config
from src.data_loader import load_clean_reduced
from src.fairness import fairness_per_race_kanon, loss_disparity
from src.kanon import evaluate_kanon
from src.plotting import plot_fairness_curves, plot_kanon_degradation
from src.preprocessing import stratified_split


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    df_train_raw = df.loc[X_train.index].copy()
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_train_raw[column] = df_train_raw[column].astype(str)
        df_test_raw[column] = df_test_raw[column].astype(str)

    # Tabla de utilidad por modelo y nivel de k
    df_baseline = pd.read_csv(config.RESULTS_DIR / "baseline.csv")
    df_baseline = df_baseline.assign(pct_suprimido=0,
                                       filas_efectivas=len(X_train),
                                       columnas_OHE=X_train.shape[1])
    results = [df_baseline]

    for k in config.K_VALUES:
        filepath = config.DATA_DIR / f"arx_output_k{k}.csv"
        if filepath.exists():
            results.append(evaluate_kanon(filepath, k, df_test_raw))
        else:
            print(f"Aviso: no se encuentra {filepath}, se omite k={k}")

    df_full = pd.concat(results, ignore_index=True)
    df_full.to_csv(config.RESULTS_DIR / "resultados_kanon.csv", index=False)
    plot_kanon_degradation(df_full, config.RESULTS_DIR / "comparativa_kanon.png")
    print("Tabla de utilidad guardada en resultados_kanon.csv")

    # Análisis de fairness por raza
    fairness_partes = [fairness_per_race_kanon(None, 0, df_train_raw, df_test_raw)]
    for k in config.K_VALUES:
        filepath = config.DATA_DIR / f"arx_output_k{k}.csv"
        if filepath.exists():
            fairness_partes.append(fairness_per_race_kanon(filepath, k, df_train_raw, df_test_raw))

    df_fairness = pd.concat(fairness_partes, ignore_index=True)
    df_fairness.to_csv(config.RESULTS_DIR / "fairness_kanon.csv", index=False)
    plot_fairness_curves(
        df_fairness,
        x_column="k",
        x_label="k",
        output_path=config.RESULTS_DIR / "fairness_kanon.png",
        title="Disparidad por subgrupo race a lo largo del barrido de k",
    )

    pivot = df_fairness.pivot_table(index="race", columns="k", values="Accuracy", aggfunc="mean")
    print("\nLoss Disparity por k:")
    for k_value in pivot.columns:
        print(f"  k={k_value}: {loss_disparity(pivot[k_value]):.4f}")


if __name__ == "__main__":
    main()
