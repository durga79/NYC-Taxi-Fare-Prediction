"""
Evaluator Module
================

Utilities for evaluating ML models on NYC Taxi data.
All implementations are original, written from PySpark MLlib documentation.
"""

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, when, count, avg, stddev
from typing import Dict, List, Any
import pandas as pd


def create_regression_evaluators(
    label_col: str = "fare_amount",
    prediction_col: str = "prediction"
) -> Dict[str, RegressionEvaluator]:
    """
    Create regression evaluators for multiple metrics.
    
    Original implementation for comprehensive evaluation.
    
    Args:
        label_col: Label column name
        prediction_col: Prediction column name
        
    Returns:
        Dictionary of evaluators
    """
    evaluators = {
        'rmse': RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName='rmse'
        ),
        'mae': RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName='mae'
        ),
        'r2': RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName='r2'
        ),
        'mse': RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName='mse'
        )
    }
    
    return evaluators


def evaluate_model(
    model: Any,
    test_df: DataFrame,
    label_col: str = "fare_amount"
) -> Dict[str, float]:
    """
    Evaluate model on test data with multiple metrics.
    
    Original implementation for model evaluation.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        label_col: Label column name
        
    Returns:
        Dictionary of metric values
    """
    # Make predictions
    predictions = model.transform(test_df)
    
    # Create evaluators
    evaluators = create_regression_evaluators(label_col=label_col)
    
    # Compute metrics
    metrics = {}
    for metric_name, evaluator in evaluators.items():
        metrics[metric_name] = float(evaluator.evaluate(predictions))
    
    return metrics


def compute_residuals(predictions_df: DataFrame, label_col: str = "fare_amount") -> DataFrame:
    """
    Compute residuals and residual statistics.
    
    Original implementation for residual analysis.
    
    Args:
        predictions_df: DataFrame with predictions
        label_col: Label column name
        
    Returns:
        DataFrame with residuals
    """
    df_residuals = predictions_df.withColumn(
        "residual",
        col(label_col) - col("prediction")
    ).withColumn(
        "abs_residual",
        spark_abs(col("residual"))
    ).withColumn(
        "percent_error",
        when(col(label_col) != 0, 
             (spark_abs(col("residual")) / spark_abs(col(label_col))) * 100.0)
        .otherwise(0.0)
    )
    
    return df_residuals


def get_residual_statistics(residuals_df: DataFrame) -> Dict[str, float]:
    """
    Compute residual statistics.
    
    Original implementation for residual analysis.
    
    Args:
        residuals_df: DataFrame with residuals
        
    Returns:
        Dictionary of residual statistics
    """
    stats = residuals_df.select(
        avg("residual").alias("mean_residual"),
        stddev("residual").alias("std_residual"),
        avg("abs_residual").alias("mean_abs_residual"),
        avg("percent_error").alias("mean_percent_error")
    ).collect()[0]
    
    residual_stats = {
        'mean_residual': float(stats['mean_residual']),
        'std_residual': float(stats['std_residual']),
        'mean_abs_residual': float(stats['mean_abs_residual']),
        'mean_percent_error': float(stats['mean_percent_error'])
    }
    
    return residual_stats


def analyze_prediction_ranges(
    predictions_df: DataFrame,
    label_col: str = "fare_amount",
    num_bins: int = 10
) -> DataFrame:
    """
    Analyze prediction accuracy across different fare ranges.
    
    Original implementation for range-based analysis.
    
    Args:
        predictions_df: DataFrame with predictions
        label_col: Label column name
        num_bins: Number of bins for analysis
        
    Returns:
        DataFrame with range-based metrics
    """
    from pyspark.sql.functions import floor, min as spark_min, max as spark_max
    
    # Get min and max for binning
    min_max = predictions_df.select(
        spark_min(label_col).alias("min_val"),
        spark_max(label_col).alias("max_val")
    ).collect()[0]
    
    min_val = float(min_max['min_val'])
    max_val = float(min_max['max_val'])
    bin_width = (max_val - min_val) / num_bins
    
    # Create bins
    df_binned = predictions_df.withColumn(
        "fare_bin",
        floor((col(label_col) - min_val) / bin_width)
    )
    
    # Compute metrics per bin
    range_analysis = df_binned.groupBy("fare_bin").agg(
        count("*").alias("count"),
        avg(label_col).alias("avg_actual"),
        avg("prediction").alias("avg_predicted"),
        avg(spark_abs(col(label_col) - col("prediction"))).alias("avg_abs_error")
    ).orderBy("fare_bin")
    
    return range_analysis


def compare_models(
    models: Dict[str, Any],
    test_df: DataFrame,
    label_col: str = "fare_amount"
) -> DataFrame:
    """
    Compare multiple models on test data.
    
    Original implementation for model comparison.
    
    Args:
        models: Dictionary of model name -> trained model
        test_df: Test DataFrame
        label_col: Label column name
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, test_df, label_col)
        
        result = {
            'model_name': model_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'mse': metrics['mse']
        }
        results.append(result)
    
    # Convert to DataFrame
    spark = test_df.sparkSession
    comparison_df = spark.createDataFrame(results)
    
    return comparison_df


def extract_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Original implementation for feature importance analysis.
    
    Args:
        model: Trained tree-based model (RF, GBT, DT)
        feature_names: List of feature names
        
    Returns:
        Pandas DataFrame with feature importance
    """
    try:
        # Get feature importances
        importances = model.featureImportances.toArray()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except AttributeError:
        print("Model does not have feature importances (likely linear model)")
        return pd.DataFrame()


def create_prediction_summary(
    predictions_df: DataFrame,
    label_col: str = "fare_amount",
    sample_size: int = 1000
) -> pd.DataFrame:
    """
    Create summary of predictions for visualization.
    
    Original implementation for Tableau export.
    
    Args:
        predictions_df: DataFrame with predictions
        label_col: Label column name
        sample_size: Number of samples to return
        
    Returns:
        Pandas DataFrame with prediction summary
    """
    # Sample for visualization
    sample_df = predictions_df.select(
        label_col,
        "prediction",
        "residual",
        "abs_residual",
        "percent_error"
    ).sample(False, min(sample_size / predictions_df.count(), 1.0), seed=42)
    
    # Convert to Pandas
    summary_pdf = sample_df.toPandas()
    
    return summary_pdf


def calculate_mape(predictions_df: DataFrame, label_col: str = "fare_amount") -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Original implementation for MAPE calculation.
    
    Args:
        predictions_df: DataFrame with predictions
        label_col: Label column name
        
    Returns:
        MAPE value
    """
    mape_df = predictions_df.withColumn(
        "ape",
        when(col(label_col) != 0,
             spark_abs((col(label_col) - col("prediction")) / col(label_col)) * 100.0)
        .otherwise(0.0)
    )
    
    mape = float(mape_df.select(avg("ape")).collect()[0][0])
    
    return mape


def evaluate_by_segment(
    predictions_df: DataFrame,
    segment_col: str,
    label_col: str = "fare_amount"
) -> DataFrame:
    """
    Evaluate model performance by segment.
    
    Original implementation for segment-based evaluation.
    
    Args:
        predictions_df: DataFrame with predictions
        segment_col: Column to segment by (e.g., 'is_rush_hour')
        label_col: Label column name
        
    Returns:
        DataFrame with segment-based metrics
    """
    segment_metrics = predictions_df.groupBy(segment_col).agg(
        count("*").alias("count"),
        avg(label_col).alias("avg_actual"),
        avg("prediction").alias("avg_predicted"),
        avg(spark_abs(col(label_col) - col("prediction"))).alias("mae"),
        avg(((col(label_col) - col("prediction")) ** 2)).alias("mse")
    )
    
    # Add RMSE
    from pyspark.sql.functions import sqrt
    segment_metrics = segment_metrics.withColumn("rmse", sqrt(col("mse")))
    
    return segment_metrics


def create_evaluation_report(
    model_name: str,
    metrics: Dict[str, float],
    residual_stats: Dict[str, float],
    feature_importance: pd.DataFrame = None
) -> Dict:
    """
    Create comprehensive evaluation report.
    
    Original implementation for reporting.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        residual_stats: Dictionary of residual statistics
        feature_importance: Feature importance DataFrame (optional)
        
    Returns:
        Evaluation report dictionary
    """
    report = {
        'model_name': model_name,
        'metrics': metrics,
        'residual_statistics': residual_stats
    }
    
    if feature_importance is not None and not feature_importance.empty:
        report['top_features'] = feature_importance.head(10).to_dict('records')
    
    return report


def save_evaluation_results(
    results: Dict,
    output_path: str
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to: {output_path}")
