"""Evaluation helpers: metrics, curves, persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config

MetricDict = Dict[str, float]


def _safe_scores(y_true: np.ndarray, y_score: np.ndarray | None, scorer) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    return scorer(y_true, y_score)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> MetricDict:
    """Compute the core classification metrics for a single run."""
    metrics: MetricDict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    roc_auc = _safe_scores(y_true, y_score, roc_auc_score)
    pr_auc = _safe_scores(y_true, y_score, average_precision_score)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    if pr_auc is not None:
        metrics["pr_auc"] = pr_auc
    return metrics


def format_metrics_with_stats(metric_values: Dict[str, List[float]]) -> MetricDict:
    """Format metric dict into mean/std schema."""
    summary: MetricDict = {}
    for metric, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        summary[f"{metric}_mean"] = float(np.mean(arr))
        summary[f"{metric}_std"] = float(np.std(arr))
    return summary


def single_run_to_summary(metrics: MetricDict) -> MetricDict:
    """Wrap single run metrics inside the mean/std convention."""
    return {f"{k}_mean": float(v) for k, v in metrics.items()} | {f"{k}_std": 0.0 for k in metrics}


def save_metrics_table(records: List[Dict[str, float]], output_path: Path) -> None:
    """Persist a list of metric dicts as CSV."""
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
    output_path: Path,
) -> None:
    """Plot absolute and normalized confusion matrices with summary metrics."""
    if y_true is None or y_pred is None:
        return
    label_list = list(labels)
    if not label_list:
        label_list = list(np.unique(np.concatenate([y_true, y_pred])))
    cm_abs = confusion_matrix(y_true, y_pred, labels=label_list)
    cm_norm = confusion_matrix(y_true, y_pred, labels=label_list, normalize="true")
    metrics = compute_classification_metrics(y_true, y_pred, None)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.7], wspace=0.25)
    ax_abs = fig.add_subplot(gs[0, 0])
    ax_norm = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    sns.heatmap(
        cm_abs,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_abs,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    ax_abs.set_title("Confusion Matrix (counts)")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        ax=ax_norm,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    ax_norm.set_title("Confusion Matrix (normalized)")
    for ax in (ax_abs, ax_norm):
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(np.arange(len(label_list)) + 0.5)
        ax.set_yticks(np.arange(len(label_list)) + 0.5)
        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)

    ax_text.axis("off")
    text_lines = [
        "Key metrics:",
        f"Accuracy: {metrics.get('accuracy', float('nan')):.3f}",
        f"Precision (macro): {metrics.get('precision_macro', float('nan')):.3f}",
        f"Recall (macro): {metrics.get('recall_macro', float('nan')):.3f}",
        f"F1 (macro): {metrics.get('f1_macro', float('nan')):.3f}",
    ]
    ax_text.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    fig.suptitle("Confusion Matrix Overview", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    try:
        html_path = output_path.with_suffix(".html")
        figly = make_subplots(rows=1, cols=2, subplot_titles=("Counts", "Normalized"))
        figly.add_trace(
            go.Heatmap(
                z=cm_abs,
                x=label_list,
                y=label_list,
                colorscale="Blues",
                showscale=False,
                text=cm_abs,
                texttemplate="%{text}",
            ),
            row=1,
            col=1,
        )
        figly.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=label_list,
                y=label_list,
                colorscale="Greens",
                showscale=True,
                text=np.round(cm_norm, 2),
                texttemplate="%{text}",
            ),
            row=1,
            col=2,
        )
        figly.update_xaxes(title_text="Predicted", row=1, col=1)
        figly.update_yaxes(title_text="Actual", row=1, col=1)
        figly.update_layout(
            title="Confusion Matrix Overview",
            width=900,
            height=450,
            template="plotly_white",
        )
        figly.write_html(str(html_path))
    except Exception:
        pass



def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, label: str, output_path: Path) -> None:
    if y_true is None or y_score is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = int(np.argmax(gmeans))
    best_fpr, best_tpr = fpr[ix], tpr[ix]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2.8, color="#1f77b4")
    plt.fill_between(fpr, tpr, alpha=0.15, color="#1f77b4")
    plt.scatter([best_fpr], [best_tpr], color="#d62728", s=60, zorder=5, label="Best G-Mean")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.25, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.text(
        0.6,
        0.2,
        f"AUC = {auc:.3f}\nBest FPR={best_fpr:.2f}\nBest TPR={best_tpr:.2f}",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    try:
        html_path = output_path.with_suffix(".html")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} (AUC={auc:.3f})",
                line=dict(width=3, color="#1f77b4"),
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.2)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best_fpr],
                y=[best_tpr],
                mode="markers",
                marker=dict(color="#d62728", size=10),
                name="Best G-Mean",
            )
        )
        fig.update_layout(
            title=f"ROC Curve - {label}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=800,
            height=600,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.write_html(str(html_path))
    except Exception:
        pass



def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, label: str, output_path: Path) -> None:
    if y_true is None or y_score is None:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true)) if len(y_true) else 0.0
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"{label} (AP={auc:.3f})", linewidth=2.8, color="#2ca02c")
    plt.fill_between(recall, precision, alpha=0.15, color="#2ca02c")
    plt.hlines(baseline, xmin=0, xmax=1, colors="gray", linestyles="dashed", label=f"Baseline={baseline:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {label}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    try:
        html_path = output_path.with_suffix(".html")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{label} (AP={auc:.3f})",
                line=dict(width=3, color="#2ca02c"),
                fill="tozeroy",
                fillcolor="rgba(44,160,44,0.2)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                name=f"Baseline={baseline:.2f}",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig.update_layout(
            title=f"Precision-Recall Curve - {label}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=800,
            height=600,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.write_html(str(html_path))
    except Exception:
        pass



def save_model(estimator, path: Path) -> None:
    """Persist an estimator with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)


def extract_probabilities(estimator, X) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return predictions and probability/score vector if available."""
    y_pred = estimator.predict(X)
    y_score = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2:
            y_score = proba[:, -1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    return y_pred, y_score
