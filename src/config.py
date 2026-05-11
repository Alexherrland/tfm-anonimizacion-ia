"""Constantes y parámetros del pipeline experimental."""

from pathlib import Path


# Rutas relativas al raíz del repositorio
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "arx_kit"
HIERARCHIES_DIR = DATA_DIR / "arx_hierarchies"
RESULTS_DIR = ROOT / "results"

# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Barridos de privacidad
K_VALUES = [2, 5, 10, 25, 50]
EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
N_REPETITIONS_DP = 5

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
