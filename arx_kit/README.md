# arx_kit — Artefactos experimentales del TFM

Carpeta organizada en tres bloques: entradas a ARX, salidas anonimizadas y resultados de los experimentos Python.

```
arx_kit/
├── inputs/                          # Entradas a ARX Desktop
│   ├── arx_train.csv                # 78.442 registros para anonimizar
│   ├── arx_test.csv                 # 19.611 registros en claro (test set)
│   ├── arx_hierarchies/             # Jerarquías de generalización por QID
│   ├── GUIA_ARX_PASO_A_PASO.md      # Procedimiento manual de la GUI
│   └── tfm-arx_k2.deid              # Proyecto ARX guardado (k=2 ejemplo)
│
├── arx_outputs/                     # CSVs anonimizados exportados desde ARX
│   ├── arx_output_k{2,5,10,25,50}.csv             # k-anonimidad pura
│   ├── arx_output_k5_l{2,3,5}.csv                 # l-diversidad sobre k=5
│   ├── arx_output_k5_t{02,025,03,035,04,05}.csv   # t-closeness sobre k=5
│   └── arx_output_k5_l*_t*.csv                    # Combinaciones triples
│
└── results/                         # Outputs de experimentos Python
    ├── kanon/                       # Fase ARX (k-anonimidad)
    │   ├── resultados_kanon.csv
    │   ├── fairness_kanon.csv
    │   └── mcnemar_kanon.csv
    ├── dp/                          # Fase Privacidad Diferencial (n=20)
    │   ├── resultados_dp.csv
    │   ├── resultados_combo.csv
    │   ├── fairness_dp.csv
    │   ├── fairness_combo.csv
    │   └── wilcoxon_dp.csv
    ├── ldiv_tclos/                  # Extensión l-diversidad / t-closeness
    │   ├── resultados_ldiv_tclos.csv
    │   ├── fairness_ldiv_tclos.csv
    │   └── mcnemar_ldiv_tclos.csv
    └── mia/                         # Auditoría MIA Black-Box
        ├── mia_results.csv
        └── mia_ldiv_tclos.csv
```

## Convención de nombres

- `arx_output_kK.csv` → k-anonimidad con valor K
- `arx_output_kK_lL.csv` → k-anonimidad + l-diversidad (l=L)
- `arx_output_kK_tNN.csv` → k-anonimidad + t-closeness (t=0.NN, p.ej. t02 = 0.2)
- `arx_output_kK_lL_tNN.csv` → triple combinación
