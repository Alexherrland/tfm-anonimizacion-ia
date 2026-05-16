"""Fase 9 — Auditoría MIA Black-Box sobre las configuraciones l-div / t-clos más estrictas.

Audita cuatro escenarios sobre Regresión Logística:
  - k=5 (referencia)
  - k=5 + l=5 (l-diversidad más estricta viable)
  - k=5 + t=0.25 (t-closeness más estricta viable)
  - k=5 + l=5 + t=0.3 (Hard, único triple distinguible de su single t)

Resultado: tabla mia_ldiv_tclos.csv con TPR, FPR y Advantage por configuración.

Nota de reproducibilidad: se utiliza un RNG independiente por configuración
(`np.random.default_rng(seed)`), con la semilla fijada explícitamente a
partir de `config.DP_SEEDS`. Esto elimina la dependencia del orden de las
configuraciones que tenía la versión inicial basada en `np.random.seed`
global, y permite añadir o reordenar configuraciones sin alterar los
muestreos de las demás.

Uso:
    python scripts/09_mia_ldiv_tclos.py
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from src import config
from src.data_loader import load_clean_reduced
from src.preprocessing import binarize_target, joint_ohe, stratified_split


def _arrays_from_anon(filepath, df_test_raw):
    df_anon = pd.read_csv(filepath, sep=";")
    df_anon = df_anon[~df_anon["race"].astype(str).str.fullmatch(r"\*")]
    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])
    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    scaler = StandardScaler().fit(X_anon_array)
    return (
        scaler.transform(X_anon_array).astype(np.float32),
        y_anon,
        scaler.transform(X_test_array).astype(np.float32),
    )


def _mia_attack(X_train, y_train, X_test, y_test, seed: int, n_attack: int = 2000):
    """Ejecuta el ataque MIA Black-Box con un RNG local sembrado por `seed`."""
    target = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    target.fit(X_train, y_train)
    art_classifier = ScikitlearnLogisticRegression(
        model=target, clip_values=(X_train.min(), X_train.max())
    )

    rng = np.random.default_rng(seed)
    n_attack = min(n_attack, len(X_test))
    attack_idx = rng.choice(len(X_train), n_attack, replace=False)
    test_idx = rng.choice(len(X_test), n_attack, replace=False)
    attack = MembershipInferenceBlackBox(
        estimator=art_classifier, attack_model_type="rf"
    )

    half = n_attack // 2
    attack.fit(
        X_train[attack_idx[:half]], y_train[attack_idx[:half]],
        X_test[test_idx[:half]], y_test[test_idx[:half]],
    )
    inf_members = attack.infer(X_train[attack_idx[half:]], y_train[attack_idx[half:]])
    inf_non_members = attack.infer(X_test[test_idx[half:]], y_test[test_idx[half:]])

    # AUC del atacante a partir de las probabilidades por ejemplo
    proba_m = np.asarray(
        attack.infer(X_train[attack_idx[half:]], y_train[attack_idx[half:]], probabilities=True)
    ).reshape(-1)
    proba_n = np.asarray(
        attack.infer(X_test[test_idx[half:]], y_test[test_idx[half:]], probabilities=True)
    ).reshape(-1)
    # Si vuelve (n,2) tras reshape, descartar primera mitad (clase 0)
    if len(proba_m) == 2 * half:
        proba_m = proba_m.reshape(half, 2)[:, -1]
        proba_n = proba_n.reshape(half, 2)[:, -1]
    scores = np.concatenate([proba_m, proba_n])
    labels = np.concatenate([np.ones(half, dtype=int), np.zeros(half, dtype=int)])
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")

    tpr = float(inf_members.mean())
    fpr = float(inf_non_members.mean())
    return tpr, fpr, tpr - fpr, auc


def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.RESULTS_KANON_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DP_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_LDIV_TCLOS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_MIA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    df_test_raw = df.loc[X_test.index].copy()
    for column in ("admission_type_id", "time_in_hospital"):
        df_test_raw[column] = df_test_raw[column].astype(str)
    y_test_array = y_test.values

    targets = [
        ("k=5 (referencia)",          "arx_output_k5.csv"),
        ("k=5 + l=5",                 "arx_output_k5_l5.csv"),
        ("k=5 + t=0.25",              "arx_output_k5_t025.csv"),
        ("k=5 + l=5 + t=0.3 (Hard)",  "arx_output_k5_l5_t03.csv"),
    ]

    results: List[dict] = []
    for i, (label, filename) in enumerate(targets):
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite")
            continue
        Xa, ya, Xt = _arrays_from_anon(filepath, df_test_raw)
        seed = config.DP_SEEDS[i % len(config.DP_SEEDS)]
        tpr, fpr, advantage, auc = _mia_attack(Xa, ya, Xt, y_test_array, seed=seed)
        results.append({
            "escenario": label,
            "archivo": filename,
            "seed": seed,
            "TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
            "Advantage": round(advantage, 4),
            "AUC": round(auc, 4),
        })
        print(f"  ✓ {label} (seed={seed})")

    df_mia = pd.DataFrame(results)
    df_mia.to_csv(config.RESULTS_MIA_DIR / "mia_ldiv_tclos.csv", index=False)
    print(df_mia.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
