"""Fase 2 — Exporta los CSVs y jerarquías para ARX Desktop.

Genera arx_kit/inputs/arx_train.csv, arx_kit/inputs/arx_test.csv y los
cinco archivos de jerarquía en arx_kit/inputs/arx_hierarchies/. El paso
de anonimización se realiza manualmente en ARX Desktop según la guía
descrita en la memoria; los CSV resultantes deben guardarse en
arx_kit/arx_outputs/.

Uso:
    python scripts/02_export_arx.py
"""

from src import config
from src.data_loader import load_clean_reduced
from src.kanon import export_for_arx, export_hierarchies
from src.preprocessing import stratified_split


def main() -> None:
    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)

    export_for_arx(
        df_reduced=df,
        train_index=X_train.index,
        test_index=X_test.index,
        output_dir=config.INPUTS_DIR,
    )
    export_hierarchies(output_dir=config.HIERARCHIES_DIR)

    print(f"CSVs exportados a {config.INPUTS_DIR}")
    print(f"Jerarquías exportadas a {config.HIERARCHIES_DIR}")
    print("\nSiguiente paso: importar arx_train.csv en ARX Desktop, cargar")
    print("las jerarquías, ejecutar el barrido k ∈ {2, 5, 10, 25, 50} y")
    print(f"exportar cada resultado como {config.ARX_OUTPUTS_DIR}/arx_output_k<k>.csv (sep=';').")


if __name__ == "__main__":
    main()
