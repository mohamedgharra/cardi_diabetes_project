"""Model registry and parameter spaces."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from . import config

ModelSpec = Tuple[object, Dict[str, list]]


def get_models(seed: int = config.SEED) -> Dict[str, ModelSpec]:
    """Return the dictionary of candidate models and their search spaces."""
    log_reg = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    grad_boost = GradientBoostingClassifier(random_state=seed)

    svc = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )

    linear_svc = LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)

    hist_gb = HistGradientBoostingClassifier(random_state=seed, max_depth=None)

    model_space: Dict[str, ModelSpec] = {
        "logistic_regression": (
            log_reg,
            {
                "model__C": np.logspace(-2, 1, 10).tolist(),
            },
        ),
        "random_forest": (
            random_forest,
            {
                "model__n_estimators": [300, 400, 500, 600],
                "model__max_depth": [None, 5, 8, 12],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        ),
        "gradient_boosting": (
            grad_boost,
            {
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [2, 3, 4],
            },
        ),
        "svc": (
            svc,
            {
                "model__C": np.logspace(-2, 2, 12).tolist(),
                "model__gamma": np.logspace(-3, 0, 10).tolist(),
            },
        ),
        "linear_svc": (
            linear_svc,
            {
                "model__C": np.logspace(-2, 1, 10).tolist(),
            },
        ),
        "hist_gradient_boosting": (
            hist_gb,
            {
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [None, 3, 5, 7],
                "model__max_iter": [200, 300, 400],
            },
        ),
    }

    return model_space
