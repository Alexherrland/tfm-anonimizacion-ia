# Carpeta `arx_kit`

Aloja los artefactos consumidos por ARX Desktop durante la fase de
$k$-anonimidad.

## Contenido

- `arx_hierarchies/`: jerarquías de generalización para los cinco
  cuasi-identificadores. Se generan automáticamente con
  `python -m scripts.02_export_arx`.
- `arx_train.csv` y `arx_test.csv`: conjuntos exportados al formato ARX
  (separador `;`). También generados por el mismo script.
- `arx_output_k<k>.csv`: salidas de ARX Desktop, una por cada nivel de
  $k$ del barrido $\{2, 5, 10, 25, 50\}$. **No se versionan en el repositorio
  por su tamaño**; el procedimiento manual para producirlos se describe
  en `docs/guia_arx.md`.

## Flujo

1. Generar los inputs con `python -m scripts.02_export_arx`.
2. Importar `arx_train.csv` en ARX Desktop (separador `;`).
3. Cargar las jerarquías sobre cada QID; marcar `readmitted` como
   *Insensitive* y los 40 atributos restantes también como *Insensitive*.
4. Ejecutar el barrido $k \in \{2, 5, 10, 25, 50\}$ y exportar cada
   resultado como `arx_output_k<k>.csv` en esta misma carpeta.
5. Reanudar el pipeline con `python -m scripts.03_evaluate_kanon`.
