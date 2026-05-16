"""Fase 8 — Evaluación de l-diversidad, t-closeness y combinaciones triples.

Lee los nueve CSVs anonimizados exportados manualmente desde ARX Desktop
(con `diag_1_category` marcado como Sensitive y los mismos QIDs de la
fase de k-anonimidad). Para cada uno entrena los cuatro modelos baseline,
calcula utilidad sobre el conjunto de test, equidad por subgrupos race y
verifica las restricciones de privacidad declaradas.

Cubre los nombres de archivo:
    arx_output_k5.csv              (referencia, k=5 solo)
    arx_output_k5_l{2,3,5}.csv     (sweep l-diversidad)
    arx_output_k5_t{05,04,035,03,025}.csv  (sweep t-closeness)
    arx_output_k5_l2_t05.csv       (triple Soft)
    arx_output_k5_l3_t035.csv      (triple Medium)
    arx_output_k5_l5_t03.csv       (triple Hard)
    arx_output_k5_l5_t025.csv      (triple Extreme)

Uso:
    python scripts/08_eval_ldiv_tclos.py
"""

from typing import List, Tuple

import pandas as pd

from src import config
from src.data_loader import load_clean_reduced
from src.ldiv_tclos import (
    evaluate_ldiv_tclos,
    fairness_ldiv_tclos,
    verify_constraints,
)
from src.preprocessing import stratified_split


# (archivo, k, l, t, tipo)
CONFIGURATIONS: List[Tuple[str, int, int, float, str]] = [
    ("arx_output_k5.csv",          5, None, None, "k_solo"),
    ("arx_output_k5_l2.csv",       5, 2,    None, "l_div"),
    ("arx_output_k5_l3.csv",       5, 3,    None, "l_div"),
    ("arx_output_k5_l5.csv",       5, 5,    None, "l_div"),
    ("arx_output_k5_t05.csv",      5, None, 0.5,  "t_clos"),
    ("arx_output_k5_t04.csv",      5, None, 0.4,  "t_clos"),
    ("arx_output_k5_t035.csv",     5, None, 0.35, "t_clos"),
    ("arx_output_k5_t03.csv",      5, None, 0.3,  "t_clos"),
    ("arx_output_k5_t025.csv",     5, None, 0.25, "t_clos"),
    ("arx_output_k5_l2_t05.csv",   5, 2,    0.5,  "triple_soft"),
    ("arx_output_k5_l3_t035.csv",  5, 3,    0.35, "triple_medium"),
    ("arx_output_k5_l5_t03.csv",   5, 5,    0.3,  "triple_hard"),
    ("arx_output_k5_l5_t025.csv",  5, 5,    0.25, "triple_extreme"),
]


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_KANON_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DP_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_LDIV_TCLOS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_MIA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    df_train_raw = df.loc[X_train.index].copy()
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_train_raw[column] = df_train_raw[column].astype(str)
        df_test_raw[column] = df_test_raw[column].astype(str)

    utility_parts = []
    fairness_parts = [fairness_ldiv_tclos(
        None, df_train_raw, df_test_raw,
        {"k": 5, "l": None, "t": None, "tipo": "k_solo", "archivo": "baseline"},
    )]
    verifications = []

    for filename, k, l, t, tipo in CONFIGURATIONS:
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite")
            continue
        tag = {"k": k, "l": l, "t": t, "tipo": tipo, "archivo": filename}
        utility_parts.append(evaluate_ldiv_tclos(filepath, df_test_raw, tag))
        fairness_parts.append(fairness_ldiv_tclos(
            filepath, df_train_raw, df_test_raw, tag,
        ))
        verifications.append(verify_constraints(
            filepath, config.QID_COLUMNS, df_train_raw,
            expected_k=k, expected_l=l, expected_t=t,
        ))
        print(f"  ✓ {filename}")

    df_utility = pd.concat(utility_parts, ignore_index=True)
    df_utility.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "resultados_ldiv_tclos.csv", index=False)
    df_fairness = pd.concat(fairness_parts, ignore_index=True)
    df_fairness.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "fairness_ldiv_tclos.csv", index=False)
    pd.DataFrame(verifications).to_csv(
        config.RESULTS_LDIV_TCLOS_DIR / "verificacion_ldiv_tclos.csv", index=False
    )
    print("\nGuardados:")
    print(f"  - resultados_ldiv_tclos.csv ({len(df_utility)} filas)")
    print(f"  - fairness_ldiv_tclos.csv ({len(df_fairness)} filas)")
    print(f"  - verificacion_ldiv_tclos.csv ({len(verifications)} filas)")


if __name__ == "__main__":
    main()
