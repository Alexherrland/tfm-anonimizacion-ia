"""Fase 5 — Defensa en profundidad: DP aplicada sobre k=10.

Replica el sweep ε ∈ {0.1, 1, 10} sobre el conjunto previamente
anonimizado con k=10 para evaluar si la combinación neutraliza el
efecto adverso de la DP sobre los grupos minoritarios.

Uso:
    python scripts/05_combo.py
"""

from src import config
from src.data_loader import load_clean_reduced
from src.differential_privacy import dp_on_kanonimized
from src.preprocessing import stratified_split


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_KANON_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DP_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_LDIV_TCLOS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_MIA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_reduced()
    _, X_test, _, _ = stratified_split(df)
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_test_raw[column] = df_test_raw[column].astype(str)

    filepath = config.ARX_OUTPUTS_DIR / "arx_output_k10.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encuentra {filepath}. Ejecuta antes la fase ARX para k=10."
        )

    df_combo = dp_on_kanonimized(
        filepath_kanon=filepath,
        df_test_raw=df_test_raw,
    )
    df_combo.to_csv(config.RESULTS_DP_DIR / "resultados_combo.csv", index=False)

    aggregated = df_combo.groupby(["modelo", "epsilon"])["Accuracy"].agg(["mean", "std"]).round(4)
    print("Resultados de la combinación k=10 + DP (Accuracy mean ± std):")
    print(aggregated.to_string())


if __name__ == "__main__":
    main()
