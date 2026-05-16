"""Constantes y parámetros del pipeline experimental."""

from pathlib import Path


# Rutas relativas al raíz del repositorio
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "arx_kit"

# Subestructura organizada de arx_kit/ (ver arx_kit/README.md)
INPUTS_DIR = DATA_DIR / "inputs"                    # arx_train, arx_test, hierarchies
ARX_OUTPUTS_DIR = DATA_DIR / "arx_outputs"          # CSVs anonimizados producidos por ARX
HIERARCHIES_DIR = INPUTS_DIR / "arx_hierarchies"
RESULTS_KANON_DIR = DATA_DIR / "results" / "kanon"
RESULTS_DP_DIR = DATA_DIR / "results" / "dp"
RESULTS_LDIV_TCLOS_DIR = DATA_DIR / "results" / "ldiv_tclos"
RESULTS_MIA_DIR = DATA_DIR / "results" / "mia"

# Compatibilidad hacia atrás: por defecto, RESULTS_DIR sigue siendo el directorio
# de resultados general (root del repo) para los scripts que aún no hayan migrado.
RESULTS_DIR = ROOT / "results"

# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Barridos de privacidad
K_VALUES = [2, 5, 10, 25, 50]
EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
N_REPETITIONS_DP = 20

# Semillas dispersas para las réplicas de Privacidad Diferencial.
# Se eligen valores no consecutivos para minimizar correlaciones residuales del
# generador pseudoaleatorio (Mersenne Twister). El número total coincide con
# N_REPETITIONS_DP.
DP_SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 6022, 8128, 9999,
            10007, 11113, 13121, 17171, 19273, 23131, 29327, 31193, 33391, 37397]

# Extensión a l-diversidad y t-closeness (k fijado en 5 por convención clínica)
K_FIXED_LDIV_TCLOS = 5
L_VALUES = [2, 3, 5]
T_VALUES = [0.5, 0.4, 0.35, 0.3, 0.25]
# Triple combinaciones (k, l, t) a lo largo del gradiente de privacidad
TRIPLE_CONFIGS = [
    (5, 2, 0.5),   # Soft
    (5, 3, 0.35),  # Medium
    (5, 5, 0.3),   # Hard
    (5, 5, 0.25),  # Extreme
]
SENSITIVE_ATTRIBUTE = "diag_1_category"

# Identificadores y atributos del dataset clínico
DATASET_ID = 296  # Diabetes 130-US hospitals (UCI ML Repo)

QID_COLUMNS = [
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "admission_type_id",
]

TARGET_COLUMN = "readmitted"

DROP_COLUMNS_FOR_ANONYMIZATION = ["diag_1", "diag_2", "diag_3", "medical_specialty"]

IMPUTE_COLUMNS = ["max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
IMPUTE_VALUE = "No_Registrado"

DROP_NA_SUBSET = ["race", "diag_1", "diag_2", "diag_3"]
