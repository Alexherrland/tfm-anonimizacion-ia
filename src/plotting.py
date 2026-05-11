"""Generación de figuras consumidas por la memoria del TFM."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def plot_kanon_degradation(df_results: pd.DataFrame, output_path: Path) -> None:
    """Curva de degradación de Accuracy y F1 a lo largo del barrido de k."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k_values_full = [0] + list(config.K_VALUES)

    for model in df_results["Modelo"].unique():
        subset = df_results[df_results["Modelo"] == model].sort_values("k")
        axes[0].plot(subset["k"], subset["Accuracy"], marker="o", label=model)
        axes[1].plot(subset["k"], subset["F1-Score"], marker="s", label=model)

    for ax, ylabel, title in [
        (axes[0], "Accuracy", "Degradación de Accuracy con k"),
        (axes[1], "F1-Score", "Degradación de F1-Score con k"),
    ]:
        ax.set_xlabel("k (privacidad sintáctica)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("symlog")
        ax.set_xticks(k_values_full)
        ax.set_xticklabels([str(k) for k in k_values_full])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_dp_degradation(df_results_aggregated: pd.DataFrame, baselines: dict, output_path: Path) -> None:
    """Curva de degradación de Accuracy/F1 vs ε con bandas de error."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epsilons = sorted(df_results_aggregated.index.get_level_values("epsilon").unique())

    for ax, mean_col, std_col, ylabel, metric_key in [
        (axes[0], "Acc_mean", "Acc_std", "Accuracy", "acc"),
        (axes[1], "F1_mean", "F1_std", "F1-Score (ponderado)", "f1"),
    ]:
        for model in ("Regresión Logística", "Naive Bayes (Gaussian)"):
            means = [df_results_aggregated.loc[(model, e), mean_col] for e in epsilons]
            stds = [df_results_aggregated.loc[(model, e), std_col] for e in epsilons]
            ax.errorbar(epsilons, means, yerr=stds, marker="o", capsize=4, label=model)

        for color, model_key in zip(("C0", "C1"), baselines.keys()):
            ax.axhline(
                baselines[model_key][0 if metric_key == "acc" else 1],
                ls="--",
                color=color,
                alpha=0.5,
                lw=1,
                label=f"Baseline {model_key.split()[0]}",
            )

        ax.set_xscale("log")
        ax.set_xticks(epsilons)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel(r"Presupuesto de privacidad $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Privacidad Diferencial: {ylabel} vs $\epsilon$ (mean$\pm$std, $n={config.N_REPETITIONS_DP}$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_fairness_curves(
    df_fairness: pd.DataFrame,
    x_column: str,
    x_label: str,
    output_path: Path,
    title: str,
) -> None:
    """Curvas por subgrupo `race` para fairness ARX o DP."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot = df_fairness.pivot_table(index="race", columns=x_column, values="Accuracy", aggfunc="mean")

    for race_value in pivot.index:
        n_subgroup = int(df_fairness[df_fairness["race"] == race_value]["n"].iloc[0])
        ax.plot(pivot.columns.tolist(), pivot.loc[race_value].values, marker="o",
                label=f"{race_value} (n={n_subgroup})")

    ax.set_xscale("symlog" if x_column == "k" else "log")
    ax.set_xticks(pivot.columns.tolist())
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (Regresión Logística)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_mia_bars(df_mia: pd.DataFrame, output_path: Path) -> None:
    """Gráfico de barras con Attack Accuracy y Advantage por escenario."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scenarios = df_mia["escenario"].tolist()
    x = range(len(scenarios))

    ax.bar([i - 0.2 for i in x], df_mia["attack_acc"], 0.4, label="Attack accuracy", color="C3")
    ax.bar([i + 0.2 for i in x], df_mia["attack_advantage"], 0.4, label="Attack advantage", color="C1")
    ax.axhline(0.5, ls="--", color="black", alpha=0.5, label="Aleatorio (Acc=0,5)")
    ax.axhline(0.0, ls=":", color="black", alpha=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylabel("Tasa")
    ax.set_title("Auditoría adversarial: MIA Black-Box sobre cada escenario")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
