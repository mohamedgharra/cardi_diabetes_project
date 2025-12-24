"""Inference script to score the unlabeled test set."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import evaluation, utils_io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer diabetes predictions on test data")
    parser.add_argument("--model", type=str, required=True, help="Path to trained joblib model")
    parser.add_argument("--data_test", type=str, required=True, help="Path to test.csv")
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Destination CSV for predictions",
    )
    return parser.parse_args()


def run_inference(model_path: str | Path, data_path: str | Path, outfile: Optional[str | Path] = None) -> pd.DataFrame:
    """Load a trained estimator, score the provided dataset, and optionally persist the output."""
    estimator = joblib.load(utils_io.resolve_path(model_path))
    test_df = utils_io.load_csv(data_path)

    y_pred, y_score = evaluation.extract_probabilities(estimator, test_df)
    if y_score is None:
        y_score = np.zeros_like(y_pred, dtype=float)

    identifiers = test_df["id"] if "id" in test_df.columns else pd.Series(range(len(test_df)), name="id")
    output = pd.DataFrame(
        {
            "id": identifiers,
            "diabetes_pred": y_pred,
            "prob_diabetes": y_score,
        }
    )
    if outfile is not None:
        utils_io.save_csv(output, outfile, index=False)
    return output


def main() -> None:
    args = parse_args()
    predictions = run_inference(args.model, args.data_test, args.outfile)
    print(f"Saved {len(predictions)} predictions to {args.outfile}")


if __name__ == "__main__":
    main()
