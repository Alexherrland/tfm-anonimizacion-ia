# TFM — Anonimización y Privacidad en IA Sanitaria

Repositorio con el código que acompaña al Trabajo Fin de Máster
*Estrategias de anonimización y privacidad en el ciclo de vida del
aprendizaje automático: un marco metodológico aplicado a la predicción
de readmisión hospitalaria*.

El estudio cuantifica el equilibrio entre privacidad y utilidad predictiva
en un caso de uso clínico, contrastando $k$-anonimidad (ARX), Privacidad
Diferencial (`diffprivlib`) y su combinación, junto con una auditoría
adversarial mediante MIA (`adversarial-robustness-toolbox`).

## Estructura del repositorio

```
.
├── src/                       Módulos del pipeline (importables como paquete)
│   ├── config.py              Constantes y rutas
│   ├── data_loader.py         Ingesta UCI, limpieza, agrupación CIE-9
│   ├── preprocessing.py       Split, OHE, escalado, percentiles para DP
│   ├── models.py              Constructores de modelos baseline y DP
│   ├── kanon.py               Soporte ARX (exportar CSV, jerarquías, evaluar)
│   ├── ldiv_tclos.py          l-diversidad, t-closeness y triples (k+l+t)
│   ├── differential_privacy.py  Sweep ε y combinación k+DP
│   ├── fairness.py            Disparidad por subgrupo race
│   ├── statistical_tests.py   McNemar y Wilcoxon
│   ├── mia.py                 Membership Inference Attack Black-Box
│   └── plotting.py            Figuras de la memoria
├── scripts/                   Orquestadores secuenciales
│   ├── 01_baseline.py
│   ├── 02_export_arx.py
│   ├── 03_evaluate_kanon.py
│   ├── 04_dp_sweep.py
│   ├── 05_combo.py
│   ├── 06_mia.py
│   ├── 07_statistical_tests.py
│   ├── 08_eval_ldiv_tclos.py
│   ├── 09_mia_ldiv_tclos.py
│   ├── 10_plots_ldiv_tclos.py
│   └── 11_mcnemar_ldiv_tclos.py
├── notebooks/
│   └── TFM_pipeline.ipynb     Notebook orquestador para Colab
├── arx_kit/                   Entradas, salidas y resultados experimentales
│   ├── inputs/                CSVs y jerarquías que consume ARX Desktop
│   │   ├── arx_train.csv
│   │   ├── arx_test.csv
│   │   ├── arx_hierarchies/   Jerarquías de generalización por QID
│   │   ├── GUIA_ARX_PASO_A_PASO.md
│   │   ├── tfm-arx.deid       Proyecto ARX de ejemplo
│   │   └── tfm-arx_k2.deid    Proyecto ARX guardado (k=2)
│   ├── arx_outputs/           CSVs anonimizados exportados desde ARX
│   └── results/               Outputs del pipeline Python
│       ├── kanon/             Resultados k-anonimidad
│       ├── dp/                Resultados Privacidad Diferencial + combo
│       ├── ldiv_tclos/        Resultados extensión l-diversidad / t-closeness
│       └── mia/               Resultados auditoría MIA
├── results/                   Figuras (PNG) de la memoria
├── docs/                      Material complementario
├── requirements.txt
├── TFM_Memory.pdf             The Memory of the TFM in pdf format
└── .gitignore
```

## Requisitos

- Python ≥ 3.10
- ARX Desktop (descargable en https://arx.deidentifier.org/) para la fase
  de $k$-anonimidad.

Las dependencias Python se instalan con:

```bash
pip install -r requirements.txt
```

## Ejecución del pipeline completo

El pipeline se compone de **once fases** que pueden ejecutarse en secuencia desde la raíz del repositorio:

```bash
export PYTHONPATH=$PWD                      # imprescindible al ejecutar por ruta
python scripts/01_baseline.py               # Fase 1: cota de utilidad
python scripts/02_export_arx.py             # Fase 2: prepara arx_kit/inputs/ para ARX
#   Ejecutar manualmente ARX Desktop con los CSV exportados (ver docs/guia_arx.md);
#   guardar cada resultado en arx_kit/arx_outputs/arx_output_k<k>.csv
python scripts/03_evaluate_kanon.py         # Fase 3: utilidad post-ARX + fairness
python scripts/04_dp_sweep.py               # Fase 4: barrido de ε
python scripts/05_combo.py                  # Fase 5: defensa en profundidad k+DP
python scripts/06_mia.py                    # Fase 6: auditoría adversarial MIA
python scripts/07_statistical_tests.py      # Fase 7: McNemar (k-anon) + Wilcoxon (DP)
#   Exportar manualmente desde ARX los 9 CSV de l-div/t-closeness/triples
#   con diag_1_category marcado como Sensitive, en arx_kit/arx_outputs/
python scripts/08_eval_ldiv_tclos.py        # Fase 8: utilidad y equidad l/t
python scripts/09_mia_ldiv_tclos.py         # Fase 9: MIA sobre configs estrictas l/t
python scripts/10_plots_ldiv_tclos.py       # Fase 10: figuras del Cap. 4
python scripts/11_mcnemar_ldiv_tclos.py     # Fase 11: McNemar sobre 4 configs extremas
```

> **Nota:** los scripts se ejecutan como ficheros (`python scripts/0X_nombre.py`) y **no** como módulos (`python -m scripts.0X_nombre`), porque los nombres empiezan por dígito y no son identificadores Python válidos. Es **imprescindible exportar `PYTHONPATH=$PWD`** (o anteponer `PYTHONPATH=.` a cada comando) desde la raíz del repo para que `from src import config` resuelva.

Cada script guarda sus resultados en `results/` (CSV con métricas y PNG
con figuras). Los nombres de los archivos coinciden con los referenciados
desde la memoria del TFM.

## Reproducibilidad

Todos los componentes estocásticos están parametrizados por la semilla
`RANDOM_STATE = 42` definida en `src/config.py`. El barrido de Privacidad
Diferencial utiliza **veinte repeticiones independientes** con semillas
dispersas (`42, 137, 271, 314, 1729, 2718, 3141, 6022, 8128, 9999, 10007,
11113, 13121, 17171, 19273, 23131, 29327, 31193, 33391, 37397`), elegidas
para minimizar correlaciones residuales del generador Mersenne Twister.
Con `n=20` el test de Wilcoxon de una muestra alcanza significancia
estadística formal frente al baseline en todas las celdas con
degradación consistente.

## Dataset

Diabetes 130-US hospitals for years 1999-2008 (UCI Machine Learning
Repository, ID 296). El conjunto se descarga automáticamente mediante
`ucimlrepo` en la primera ejecución.

## Cita

Trabajo Fin de Máster, Máster Universitario en Ingeniería y Ciencia de
Datos, UNED.
