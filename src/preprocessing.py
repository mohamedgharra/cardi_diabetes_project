"""Preprocessing utilities and sklearn pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class ColumnCleaner(BaseEstimator, TransformerMixin):
    """Subset, rename, and order columns prior to modeling."""

    feature_list: List[str]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnCleaner expects a pandas DataFrame as input")
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._to_dataframe(X)
        renamed = {col: config.normalize_column_name(col) for col in df.columns}
        df = df.rename(columns=renamed)
        df = df.drop(columns=[col for col in df.columns if col in config.EXCLUDED_FEATURES], errors="ignore")
        df = df.reindex(columns=self.feature_list, fill_value=np.nan)
        return df

    def _to_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if hasattr(self, "input_columns_"):
            return pd.DataFrame(X, columns=self.input_columns_)
        raise TypeError("ColumnCleaner must be fitted on a DataFrame before transforming other types")


@dataclass
class ZeroToNanTransformer(BaseEstimator, TransformerMixin):
    """Convert implausible zero values to NaN for specified columns."""

    columns: Iterable[str]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col].mask(df[col] == 0)
        return df


@dataclass
class FeatureWeighter(BaseEstimator, TransformerMixin):
    """Multiply a subset of features by a constant factor."""

    columns: Iterable[str]
    factor: float = 1.1

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = df[col] * self.factor
        return df


class BinaryCategoryEncoder(BaseEstimator, TransformerMixin):
    """Ensure binary categorical features are encoded as 0/1."""

    def __init__(self, column_names: List[str]):
        self.column_names = column_names

    def fit(self, X, y=None):
        self.columns_ = list(self.column_names)
        return self

    def transform(self, X):
        arr = np.asarray(X)
        df = pd.DataFrame(arr, columns=self.columns_)
        for col in df.columns:
            df[col] = df[col].apply(self._to_binary)
        return df.values

    def get_feature_names_out(self, input_features=None):
        """Return output feature names (compat for sklearn's get_feature_names_out)."""
        return np.array(self.columns_)

    @staticmethod
    def _to_binary(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return np.nan
            return int(value >= 0.5)
        lowered = str(value).strip().lower()
        if lowered in config.BINARY_TRUE_VALUES:
            return 1
        if lowered in config.BINARY_FALSE_VALUES:
            return 0
        try:
            return int(float(lowered))
        except ValueError:
            raise ValueError(f"Cannot convert value '{value}' to binary")


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature lists present in df."""
    available = {config.normalize_column_name(col) for col in df.columns}
    numeric_cols = [col for col in config.NUMERIC_FEATURES if col in available]
    categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in available]
    return numeric_cols, categorical_cols


def build_preprocess_pipeline(feature_list: List[str], weight_top_features: bool = False) -> Pipeline:
    """Compose the preprocessing pipeline requested in the spec."""
    numeric_cols = [col for col in config.NUMERIC_FEATURES if col in feature_list]
    categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in feature_list]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", BinaryCategoryEncoder(categorical_cols)),
        ]
    ) if categorical_cols else "drop"

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    steps: List[tuple] = [
        ("clean_columns", ColumnCleaner(feature_list=feature_list)),
        ("zero_to_nan", ZeroToNanTransformer(config.ZERO_AS_NAN_FEATURES)),
    ]

    if weight_top_features:
        priority_cols = [col for col in config.FEATURE_PRIORITY[:5] if col in feature_list]
        steps.append(("weight_priority", FeatureWeighter(columns=priority_cols)))

    steps.append(("column_transformer", column_transformer))

    return Pipeline(steps=steps)


def get_ordered_feature_list(df: pd.DataFrame) -> List[str]:
    """Return the project feature priority list filtered by available columns."""
    available = {config.normalize_column_name(col) for col in df.columns}
    return [col for col in config.FEATURE_PRIORITY if col in available]
