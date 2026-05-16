"""Fase 11 — Test de McNemar sobre las configuraciones extremas de l-div / t-closeness.

Replica la batería de tests McNemar de la Fase 7 sobre la selección de
configuraciones más informativas de la extensión sintáctica de la Fase 8:
- l-diversidad estricta (l=5)
- t-closeness relajado (t=0.5)
- t-closeness en la frontera (t=0.25)
- Triple Hard (k=5 + l=5 + t=0.3, única triple distinguible byte a byte
  de su contrapartida t-closeness sola)

Las nueve configuraciones restantes se omiten por redundancia con singles
ya testeados o por tener un efecto por debajo del umbral de detección útil.

Uso:
    python scripts/11_mcnemar_ldiv_tclos.py
"""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import load_clean_reduced
from src.models import build_baseline_models
from src.preprocessing import binarize_target, fit_scaler, joint_ohe, stratified_split
from src.statistical_tests import apply_bonferroni, mcnemar_test


# Selección razonada de configuraciones a testear (subconjunto de Fase 8)
CONFIGURATIONS: List[Tuple[str, str]] = [
    ("k=5 + l=5",                "arx_output_k5_l5.csv"),
    ("k=5 + t=0.5",              "arx_output_k5_t05.csv"),
    ("k=5 + t=0.25",             "arx_output_k5_t025.csv"),
    ("k=5 + l=5 + t=0.3 (Hard)", "arx_output_k5_l5_t03.csv"),
]


def _predict_on_clean(X_train_scaled, X_test_scaled, y_train) -> Dict[str, "pd.Series"]:
    """Entrena cada modelo baseline sobre el conjunto limpio y devuelve predicciones.

    El espacio de features (X_train_scaled, X_test_scaled) debe construirse de
    forma idéntica a `stratified_split + fit_scaler` (es decir, sin forzar a
    string admission_type_id ni time_in_hospital). Esto garantiza que el
    baseline empleado aquí coincide byte a byte con el de las Fases 1 y 7,
    y por tanto con la columna `k=0` de `resultados_kanon.csv`.
    """
    predictions: Dict[str, "pd.Series"] = {}
    for name, model in build_baseline_models().items():
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_test_scaled)
    return predictions


def _predict_on_anonymized(filepath, df_test_raw) -> Dict[str, "pd.Series"]:
    """Entrena cada modelo baseline sobre el conjunto anonimizado y devuelve predicciones."""
    df_anon = pd.read_csv(filepath, sep=";")
    df_anon = df_anon[~df_anon["race"].astype(str).str.fullmatch(r"\*")].copy()
    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])
    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    scaler = StandardScaler().fit(X_anon_array)
    X_anon_scaled = scaler.transform(X_anon_array)
    X_test_scaled = scaler.transform(X_test_array)
    predictions: Dict[str, "pd.Series"] = {}
    for name, model in build_baseline_models().items():
        model.fit(X_anon_scaled, y_anon)
        predictions[name] = model.predict(X_test_scaled)
    return predictions


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_KANON_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DP_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_LDIV_TCLOS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_MIA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    # Baseline (k=0): mismo espacio de features que en las Fases 1 y 7 — OHE
    # producido por stratified_split (admission_type_id/time_in_hospital como
    # int) y StandardScaler ajustado sobre el train. Esto garantiza la
    # coherencia con `resultados_kanon.csv` y con `tab:mcnemar`.
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # df_*_raw: copia con admission_type_id/time_in_hospital convertidos a
    # string, requerida por las jerarquías de ARX y por joint_ohe en las
    # configuraciones anonimizadas (compatible con la Fase 7).
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_test_raw[column] = df_test_raw[column].astype(str)
    y_test_array = y_test.values

    print("Predicciones baseline (k=0)...")
    baseline_predictions = _predict_on_clean(X_train_scaled, X_test_scaled, y_train)

    rows: List[Dict] = []
    for label, filename in CONFIGURATIONS:
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite {label}")
            continue
        print(f"  → {label}")
        anon_predictions = _predict_on_anonymized(filepath, df_test_raw)
        for model_name in baseline_predictions:
            p_value, b, c, chi_squared = mcnemar_test(
                y_test_array,
                baseline_predictions[model_name],
                anon_predictions[model_name],
            )
            rows.append({
                "configuracion": label,
                "archivo": filename,
                "modelo": model_name,
                "b": b,
                "c": c,
                "chi2": round(chi_squared, 4),
                "p_value": p_value,
                "significativo_005": p_value < 0.05,
            })

    # Aplicamos Bonferroni sobre la familia completa de tests
    # (N = nº de filas = 4 configuraciones × 4 modelos = 16) para mantener la
    # coherencia con la batería de la Fase 7 (build_mcnemar_table).
    df_results = apply_bonferroni(pd.DataFrame(rows))
    df_results.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "mcnemar_ldiv_tclos.csv", index=False)
    print(f"\n✓ mcnemar_ldiv_tclos.csv guardado con {len(df_results)} filas")
    print(df_results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
