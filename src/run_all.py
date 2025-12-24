"""Entrypoint for the full training, validation, and reporting pipeline."""
from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, evaluation, models, preprocessing, train_validate, utils_io, visualize


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value from '{value}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare diabetes models")
    parser.add_argument("--data_train", type=str, required=True)
    parser.add_argument("--data_test", type=str, required=True)
    parser.add_argument("--techniques", type=str, default="holdout,cv5,cv5_tuning")
    parser.add_argument("--weight_top_features", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def _log(message: str, callback: Optional[Callable[[str], None]]) -> None:
    if callback is not None:
        callback(message)
    else:
        print(message)


def _validate_techniques(techniques: List[str]) -> List[str]:
    valid = {"holdout", "cv5", "cv5_tuning"}
    cleaned = [tech.strip() for tech in techniques if tech.strip()]
    if not cleaned:
        raise ValueError("At least one validation technique must be provided")
    for tech in cleaned:
        if tech not in valid:
            raise ValueError(f"Unknown validation technique: {tech}")
    return cleaned


def run_pipeline(
    data_train: str | Path,
    data_test: str | Path,
    techniques: List[str] | str,
    weight_top_features: bool,
    seed: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Execute the full training workflow and return metadata about the run."""
    config.set_global_seed(seed)

    train_df = utils_io.load_csv(data_train)
    test_df = utils_io.load_csv(data_test)

    if config.TARGET not in train_df.columns:
        raise ValueError(f"Target column '{config.TARGET}' not found in training data")

    numeric_cols, categorical_cols = preprocessing.get_feature_lists(train_df)
    feature_list = numeric_cols + categorical_cols
    if not feature_list:
        raise ValueError("No valid cardiological features detected in the training data")

    preprocess_pipeline = preprocessing.build_preprocess_pipeline(
        feature_list=feature_list,
        weight_top_features=weight_top_features,
    )

    X = train_df.drop(columns=[config.TARGET])
    y = train_df[config.TARGET]

    _log(f"Training rows: {len(train_df)} | Features used: {feature_list}", progress_callback)
    _log(f"Test rows: {len(test_df)} (for later inference)", progress_callback)

    visualize.plot_feature_distributions(train_df, feature_list, config.TARGET, config.EDA_DIR)
    visualize.plot_correlation_heatmap(
        train_df[feature_list + [config.TARGET]],
        config.EDA_DIR / "correlation_heatmap.png",
    )
    visualize.plot_split_boxes(
        train_df,
        [feat for feat in ["heartRate", "BMI", "sysBP", "diaBP", "totChol"] if feat in feature_list],
        config.TARGET,
        config.EDA_DIR,
    )

    if isinstance(techniques, str):
        requested_techniques = techniques.split(",")
    else:
        requested_techniques = techniques
    requested_techniques = _validate_techniques(requested_techniques)

    model_registry = models.get_models(seed)

    all_results: List[Dict[str, float]] = []
    best_tracker = {
        "score": -np.inf,
        "f1": -np.inf,
        "model": None,
        "technique": None,
        "estimator": None,
    }
    best_per_model: Dict[str, train_validate.TechniqueResult] = {}

    for model_name, (estimator, param_space) in model_registry.items():
        _log(f"Evaluating model: {model_name}", progress_callback)
        results = train_validate.evaluate_model_with_techniques(
            model_name,
            estimator,
            preprocess_pipeline,
            param_space,
            requested_techniques,
            X,
            y,
            seed,
        )
        for result in results:
            record = {"model": model_name, "technique": result.technique}
            record.update(result.metrics)
            all_results.append(record)

            roc_auc = result.metrics.get("roc_auc_mean")
            f1_macro = result.metrics.get("f1_macro_mean")
            if roc_auc is not None:
                if roc_auc > best_tracker["score"] or (
                    np.isclose(roc_auc, best_tracker["score"]) and f1_macro is not None and f1_macro > best_tracker["f1"]
                ):
                    best_tracker.update(
                        {
                            "score": roc_auc,
                            "f1": f1_macro or -np.inf,
                            "model": model_name,
                            "technique": result.technique,
                            "estimator": result.estimator,
                        }
                    )
            current_best = best_per_model.get(model_name)
            current_score = current_best.metrics.get("roc_auc_mean") if current_best else -np.inf
            if (roc_auc or -np.inf) > (current_score or -np.inf):
                best_per_model[model_name] = result

            if result.y_true is not None and result.y_score is not None:
                base_name = f"{model_name}_{result.technique}"
                evaluation.plot_confusion_matrices(
                    result.y_true,
                    result.y_pred,
                    labels=[0, 1],
                    output_path=config.FIGS_DIR / f"{base_name}_confusion.png",
                )
                evaluation.plot_roc_curve(
                    result.y_true,
                    result.y_score,
                    label=f"{model_name} {result.technique}",
                    output_path=config.FIGS_DIR / f"{base_name}_roc.png",
                )
                evaluation.plot_pr_curve(
                    result.y_true,
                    result.y_score,
                    label=f"{model_name} {result.technique}",
                    output_path=config.FIGS_DIR / f"{base_name}_pr.png",
                )

    if not all_results:
        raise RuntimeError("No training result was produced. Check selected techniques/models.")

    metrics_path = config.TABLES_DIR / "summary_metrics.csv"
    evaluation.save_metrics_table(all_results, metrics_path)

    metrics_df = pd.DataFrame(all_results)
    if not metrics_df.empty:
        visualize.plot_model_comparison_bars(
            metrics_df,
            "accuracy",
            config.FIGS_DIR / "comparison_accuracy.png",
        )
        visualize.plot_model_comparison_bars(
            metrics_df,
            "f1_macro",
            config.FIGS_DIR / "comparison_f1.png",
        )
        visualize.plot_metric_heatmap(
            metrics_df,
            "roc_auc",
            config.FIGS_DIR / "roc_auc_heatmap.png",
        )

    best_estimator = best_tracker["estimator"]
    if best_estimator is None:
        raise RuntimeError("No model was successfully trained; cannot proceed")

    best_model_path = config.MODELS_DIR / "best_model.joblib"
    evaluation.save_model(best_estimator, best_model_path)

    important_models = {"random_forest", "gradient_boosting", "hist_gradient_boosting"}
    importance_tables = []
    for model_name in important_models:
        if model_name not in best_per_model:
            continue
        estimator = best_per_model[model_name].estimator
        csv_path = config.TABLES_DIR / f"feature_importance_{model_name}.csv"
        importance_df = visualize.plot_feature_importance(
            estimator,
            model_name,
            config.FIGS_DIR / f"feature_importance_{model_name}.png",
        )
        perm_df = visualize.compute_permutation_importance(
            estimator,
            X,
            y,
            model_name,
            config.FIGS_DIR / f"permutation_{model_name}.png",
        )
        if importance_df.empty and perm_df.empty:
            continue
        if importance_df.empty:
            importance_df = pd.DataFrame({"feature": perm_df["feature"], "importance": np.nan})
        if perm_df.empty:
            perm_df = pd.DataFrame({"feature": importance_df["feature"], "importance": np.nan})
        combined = importance_df.merge(
            perm_df,
            on="feature",
            how="outer",
            suffixes=("_model", "_permutation"),
        )
        combined.to_csv(csv_path, index=False)
        importance_tables.append((model_name, combined))

    if "logistic_regression" in best_per_model:
        estimator = best_per_model["logistic_regression"].estimator
        odds_df = visualize.plot_logistic_odds(
            estimator,
            X,
            y,
            config.FIGS_DIR / "logistic_odds.png",
        )
        if not odds_df.empty:
            odds_df.to_csv(config.TABLES_DIR / "logistic_odds.csv", index=False)

    dataset_size = len(train_df)
    missing_pct = train_df.isna().mean().mean() * 100
    techniques_str = ", ".join(requested_techniques)

    if not metrics_df.empty:
        top_models = metrics_df.sort_values("roc_auc_mean", ascending=False)[
            ["model", "technique", "roc_auc_mean", "f1_macro_mean"]
        ].head(3)
    else:
        top_models = pd.DataFrame(columns=["model", "technique", "roc_auc_mean", "f1_macro_mean"])
    top_features = []
    if importance_tables:
        avg_df = (
            pd.concat([tbl for _, tbl in importance_tables])
            .groupby("feature")["importance_model"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        top_features = avg_df.index.tolist()

    summary_lines = [
        "# Training Summary",
        f"- Dataset size: {dataset_size} rows",
        f"- Average missing percentage handled: {missing_pct:.2f} %",
        f"- Techniques used: {techniques_str}",
        f"- Best model: {best_tracker['model']} ({best_tracker['technique']}) with ROC-AUC={best_tracker['score']:.3f}",
        "- Top 3 model combinations:",
    ]
    for _, row in top_models.iterrows():
        summary_lines.append(
            f"  - {row['model']} ({row['technique']}): ROC-AUC={row['roc_auc_mean']:.3f}, F1={row['f1_macro_mean']:.3f}"
        )
    if top_features:
        summary_lines.append("- Top-10 feature ranking (media importanza):")
        for feat in top_features:
            summary_lines.append(f"  - {feat}")
    summary_lines.append("- Grafici salvati in reports/eda/ e reports/figs/")

    utils_io.write_text("\n".join(summary_lines), config.SUMMARY_PATH)
    _log("Pipeline completed successfully.", progress_callback)

    return {
        "best_model": best_tracker["model"],
        "best_technique": best_tracker["technique"],
        "best_score": best_tracker["score"],
        "metrics_path": metrics_path,
        "summary_path": config.SUMMARY_PATH,
        "best_model_path": best_model_path,
        "metrics_df": metrics_df,
    }


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_train=args.data_train,
        data_test=args.data_test,
        techniques=args.techniques,
        weight_top_features=args.weight_top_features,
        seed=args.seed,
        progress_callback=print,
    )


if __name__ == "__main__":
    main()
