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
│   └── 07_statistical_tests.py
├── notebooks/
│   └── TFM_pipeline.ipynb     Notebook orquestador para Colab
├── arx_kit/                   CSVs y jerarquías consumidos por ARX Desktop
│   └── arx_hierarchies/
├── results/                   Tablas (CSV) y figuras (PNG) generadas
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

El pipeline se compone de siete fases que pueden ejecutarse en secuencia:

```bash
python -m scripts.01_baseline               # Fase 1: cota de utilidad
python -m scripts.02_export_arx             # Fase 2: prepara entrada para ARX
#   Ejecutar manualmente ARX Desktop con los CSV exportados
#   (ver docs/guia_arx.md para el procedimiento detallado)
python -m scripts.03_evaluate_kanon         # Fase 3: utilidad post-ARX + fairness
python -m scripts.04_dp_sweep               # Fase 4: barrido de ε
python -m scripts.05_combo                  # Fase 5: defensa en profundidad
python -m scripts.06_mia                    # Fase 6: auditoría adversarial
python -m scripts.07_statistical_tests      # Fase 7: tests de significancia
```

Cada script guarda sus resultados en `results/` (CSV con métricas y PNG
con figuras). Los nombres de los archivos coinciden con los referenciados
desde la memoria del TFM.

## Reproducibilidad

Todos los componentes estocásticos están parametrizados por la semilla
`RANDOM_STATE = 42` definida en `src/config.py`. El barrido de Privacidad
Diferencial utiliza cinco repeticiones independientes con semillas
`42, 43, 44, 45, 46` para estimar medias y desviaciones típicas.

## Dataset

Diabetes 130-US hospitals for years 1999-2008 (UCI Machine Learning
Repository, ID 296). El conjunto se descarga automáticamente mediante
`ucimlrepo` en la primera ejecución.

## Cita

Trabajo Fin de Máster, Máster Universitario en Ingeniería y Ciencia de
Datos, UNED.
