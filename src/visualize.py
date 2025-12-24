"""Visualization utilities for EDA, model comparison, and feature importance."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# Use a non-interactive backend to avoid Tkinter errors in headless environments
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config

sns.set_style("whitegrid")


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Iterable[str],
    target: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            stat="density",
            common_norm=False,
            element="step",
            palette="Set1",
        )
        plt.title(f"Distribuzione di {feature}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{feature}.png", dpi=200)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Heatmap di correlazione (numeriche)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_split_boxes(df: pd.DataFrame, features: List[str], target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(5, 4))
        sns.violinplot(data=df, x=target, y=feature, palette="pastel", inner="quartile")
        plt.title(f"Distribuzione {feature} per {target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"violin_{feature}.png", dpi=200)
        plt.close()


def _get_feature_names_from_pipeline(preprocess: Pipeline) -> List[str]:
    column_transformer = preprocess.named_steps.get("column_transformer")
    if column_transformer is None:
        return []
    raw_names = column_transformer.get_feature_names_out()
    cleaned = []
    for name in raw_names:
        if "__" in name:
            cleaned.append(name.split("__", maxsplit=1)[-1])
        else:
            cleaned.append(name)
    return cleaned


def plot_feature_importance(
    estimator: Pipeline,
    model_name: str,
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    model = estimator.named_steps.get("model")
    preprocess = estimator.named_steps.get("preprocess")
    if model is None or preprocess is None:
        return pd.DataFrame()
    feature_names = _get_feature_names_from_pipeline(preprocess)
    if not feature_names:
        return pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        return pd.DataFrame()
    order = np.argsort(importances)[::-1]
    # Defensive: if importances length doesn't match feature_names, align to the min length
    n = min(len(importances), len(feature_names))
    order = order[:n]
    ordered_features = np.array(feature_names)[order]
    ordered_importances = importances[order]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=ordered_importances[:top_n], y=ordered_features[:top_n], palette="viridis")
    plt.xlabel("Importanza")
    plt.title(f"Feature importance - {model_name}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return pd.DataFrame({"feature": ordered_features, "importance": ordered_importances})


def compute_permutation_importance(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    output_path: Path,
    random_state: int = config.SEED,
) -> pd.DataFrame:
    preprocess = estimator.named_steps.get("preprocess")
    if preprocess is None:
        return pd.DataFrame()
    feature_names = _get_feature_names_from_pipeline(preprocess)
    if not feature_names:
        return pd.DataFrame()
    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=15,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importances = result.importances_mean
    order = np.argsort(importances)[::-1]
    n = min(len(importances), len(feature_names))
    # Keep only indices that exist in feature_names to avoid out-of-bounds issues
    order = [idx for idx in order if idx < len(feature_names)][:n]
    ordered_features = np.array(feature_names)[order]
    ordered_values = importances[order]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=ordered_values[:10], y=ordered_features[:10], palette="mako")
    plt.title(f"Permutation importance - {model_name}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return pd.DataFrame({"feature": ordered_features, "importance": ordered_values})


def plot_model_comparison_bars(metrics_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y=f"{metric}_mean", hue="technique", palette="Set2")
    plt.title(f"Confronto {metric}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_heatmap(metrics_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    pivot = metrics_df.pivot(index="model", columns="technique", values=f"{metric}_mean")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="rocket_r")
    plt.title(f"Heatmap {metric}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_logistic_odds(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    n_boot: int = 40,
    random_state: int = config.SEED,
) -> pd.DataFrame:
    model = estimator.named_steps.get("model")
    preprocess = estimator.named_steps.get("preprocess")
    if model is None or preprocess is None or not hasattr(model, "coef_"):
        return pd.DataFrame()
    feature_names = _get_feature_names_from_pipeline(preprocess)
    if not feature_names:
        return pd.DataFrame()
    coefs = model.coef_.ravel()

    rng = np.random.default_rng(random_state)
    bootstraps = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        boot_estimator = clone(estimator)
        boot_estimator.fit(X.iloc[idx], y.iloc[idx])
        boot_coefs = boot_estimator.named_steps["model"].coef_.ravel()
        bootstraps.append(boot_coefs)
    boot_array = np.vstack(bootstraps)
    conf_low = np.percentile(boot_array, 2.5, axis=0)
    conf_high = np.percentile(boot_array, 97.5, axis=0)

    order = np.argsort(np.abs(coefs))[::-1]
    n = min(len(coefs), len(feature_names))
    order = order[:n]
    ordered_features = np.array(feature_names)[order]
    ordered_coefs = coefs[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=ordered_coefs, y=ordered_features, palette="coolwarm", orient="h", ax=ax)
    lower_err = np.abs(ordered_coefs - conf_low[order])
    upper_err = np.abs(conf_high[order] - ordered_coefs)
    ax.errorbar(
        x=ordered_coefs,
        y=np.arange(len(ordered_features)),
        xerr=np.vstack((lower_err, upper_err)),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Logistic Regression Odds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return pd.DataFrame(
        {
            "feature": ordered_features,
            "coef": ordered_coefs,
            "ci_low": conf_low[order],
            "ci_high": conf_high[order],
        }
    )
