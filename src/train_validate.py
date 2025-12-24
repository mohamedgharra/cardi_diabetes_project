"""Training utilities implementing multiple validation strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config, evaluation

SCORERS = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "precision_weighted": "precision_weighted",
    "recall_macro": "recall_macro",
    "recall_weighted": "recall_weighted",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "roc_auc": "roc_auc",
    # Use sklearn's built-in average_precision scorer name for compatibility
    "pr_auc": "average_precision",
}


@dataclass
class TechniqueResult:
    model: str
    technique: str
    metrics: Dict[str, float]
    estimator: Pipeline
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None


def _build_pipeline(preprocess_pipeline: Pipeline, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", clone(preprocess_pipeline)),
            ("model", clone(estimator)),
        ]
    )


def run_holdout(
    model_name: str,
    estimator,
    preprocess_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = config.SEED,
) -> TechniqueResult:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    pipeline = _build_pipeline(preprocess_pipeline, estimator)
    pipeline.fit(X_train, y_train)
    y_pred, y_score = evaluation.extract_probabilities(pipeline, X_val)
    metrics = evaluation.compute_classification_metrics(y_val, y_pred, y_score)
    summary = evaluation.single_run_to_summary(metrics)
    pipeline.fit(X, y)
    return TechniqueResult(
        model=model_name,
        technique="holdout",
        metrics=summary,
        estimator=pipeline,
        y_true=y_val.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
    )


def run_cv5(
    model_name: str,
    estimator,
    preprocess_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = config.SEED,
) -> TechniqueResult:
    pipeline = _build_pipeline(preprocess_pipeline, estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipeline,
        X,
        y,
        scoring=SCORERS,
        cv=cv,
        n_jobs=-1,
        return_estimator=False,
    )
    metric_values = {
        metric: scores[f"test_{metric}"]
        for metric in [
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
            "pr_auc",
        ]
    }
    summary = evaluation.format_metrics_with_stats(metric_values)
    pipeline.fit(X, y)
    return TechniqueResult(
        model=model_name,
        technique="cv5",
        metrics=summary,
        estimator=pipeline,
    )


def _estimate_n_iter(param_distributions: Dict[str, List]) -> int:
    sizes = [len(values) for values in param_distributions.values() if hasattr(values, "__len__")]
    if not sizes:
        return 10
    total = int(np.prod(sizes))
    return max(5, min(25, total))


def run_cv5_tuning(
    model_name: str,
    estimator,
    preprocess_pipeline: Pipeline,
    param_distributions: Dict[str, List],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = config.SEED,
) -> TechniqueResult:
    pipeline = _build_pipeline(preprocess_pipeline, estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=_estimate_n_iter(param_distributions),
        scoring=SCORERS,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    search.fit(X, y)
    best_idx = search.best_index_
    summary = {}
    for metric in SCORERS.keys():
        mean_key = f"mean_test_{metric}"
        std_key = f"std_test_{metric}"
        summary[f"{metric}_mean"] = float(search.cv_results_[mean_key][best_idx])
        summary[f"{metric}_std"] = float(search.cv_results_[std_key][best_idx])
    return TechniqueResult(
        model=model_name,
        technique="cv5_tuning",
        metrics=summary,
        estimator=search.best_estimator_,
    )


def evaluate_model_with_techniques(
    model_name: str,
    estimator,
    preprocess_pipeline: Pipeline,
    param_distributions: Dict[str, List],
    techniques: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = config.SEED,
) -> List[TechniqueResult]:
    results: List[TechniqueResult] = []
    for technique in techniques:
        if technique == "holdout":
            results.append(run_holdout(model_name, estimator, preprocess_pipeline, X, y, seed))
        elif technique == "cv5":
            results.append(run_cv5(model_name, estimator, preprocess_pipeline, X, y, seed))
        elif technique == "cv5_tuning":
            if not param_distributions:
                continue
            results.append(
                run_cv5_tuning(
                    model_name,
                    estimator,
                    preprocess_pipeline,
                    param_distributions,
                    X,
                    y,
                    seed,
                )
            )
    return results
