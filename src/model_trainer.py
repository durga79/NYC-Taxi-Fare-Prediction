"""
Model Trainer Module
====================

Utilities for training ML models on NYC Taxi data.
All implementations are original, written from PySpark MLlib documentation.
"""

from pyspark.sql import DataFrame
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    GeneralizedLinearRegression
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from typing import Dict, Tuple, Any
import time


def train_linear_regression(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "fare_amount",
    **kwargs
) -> Tuple[Any, float]:
    """
    Train Linear Regression model.
    
    Original implementation for baseline regression.
    
    Args:
        train_df: Training DataFrame
        features_col: Features column name
        label_col: Label column name
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    lr = LinearRegression(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=kwargs.get('maxIter', 100),
        regParam=kwargs.get('regParam', 0.0),
        elasticNetParam=kwargs.get('elasticNetParam', 0.0)
    )
    
    start_time = time.time()
    model = lr.fit(train_df)
    train_time = time.time() - start_time
    
    return model, train_time


def train_decision_tree(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "fare_amount",
    **kwargs
) -> Tuple[Any, float]:
    """
    Train Decision Tree Regressor.
    
    Original implementation for tree-based regression.
    
    Args:
        train_df: Training DataFrame
        features_col: Features column name
        label_col: Label column name
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    dt = DecisionTreeRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        maxDepth=kwargs.get('maxDepth', 10),
        maxBins=kwargs.get('maxBins', 32),
        minInstancesPerNode=kwargs.get('minInstancesPerNode', 1),
        seed=kwargs.get('seed', 42)
    )
    
    start_time = time.time()
    model = dt.fit(train_df)
    train_time = time.time() - start_time
    
    return model, train_time


def train_random_forest(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "fare_amount",
    **kwargs
) -> Tuple[Any, float]:
    """
    Train Random Forest Regressor.
    
    Original implementation for ensemble regression.
    
    Args:
        train_df: Training DataFrame
        features_col: Features column name
        label_col: Label column name
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    rf = RandomForestRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        numTrees=kwargs.get('numTrees', 50),
        maxDepth=kwargs.get('maxDepth', 10),
        maxBins=kwargs.get('maxBins', 32),
        minInstancesPerNode=kwargs.get('minInstancesPerNode', 1),
        subsamplingRate=kwargs.get('subsamplingRate', 1.0),
        seed=kwargs.get('seed', 42)
    )
    
    start_time = time.time()
    model = rf.fit(train_df)
    train_time = time.time() - start_time
    
    return model, train_time


def train_gbt(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "fare_amount",
    **kwargs
) -> Tuple[Any, float]:
    """
    Train Gradient Boosted Trees Regressor.
    
    Original implementation for boosting regression.
    
    Args:
        train_df: Training DataFrame
        features_col: Features column name
        label_col: Label column name
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    gbt = GBTRegressor(
        featuresCol=features_col,
        labelCol=label_col,
        maxIter=kwargs.get('maxIter', 50),
        maxDepth=kwargs.get('maxDepth', 5),
        maxBins=kwargs.get('maxBins', 32),
        minInstancesPerNode=kwargs.get('minInstancesPerNode', 1),
        stepSize=kwargs.get('stepSize', 0.1),
        subsamplingRate=kwargs.get('subsamplingRate', 1.0),
        seed=kwargs.get('seed', 42)
    )
    
    start_time = time.time()
    model = gbt.fit(train_df)
    train_time = time.time() - start_time
    
    return model, train_time


def train_glr(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "fare_amount",
    **kwargs
) -> Tuple[Any, float]:
    """
    Train Generalized Linear Regression model.
    
    Original implementation for GLM regression.
    
    Args:
        train_df: Training DataFrame
        features_col: Features column name
        label_col: Label column name
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training time in seconds)
    """
    glr = GeneralizedLinearRegression(
        featuresCol=features_col,
        labelCol=label_col,
        family=kwargs.get('family', 'gaussian'),
        link=kwargs.get('link', 'identity'),
        maxIter=kwargs.get('maxIter', 100),
        regParam=kwargs.get('regParam', 0.0)
    )
    
    start_time = time.time()
    model = glr.fit(train_df)
    train_time = time.time() - start_time
    
    return model, train_time


def tune_model_with_cv(
    estimator: Any,
    param_grid: Dict,
    train_df: DataFrame,
    evaluator: RegressionEvaluator,
    num_folds: int = 3,
    parallelism: int = 4,
    seed: int = 42
) -> Tuple[Any, float]:
    """
    Tune model using CrossValidator.
    
    Original implementation for distributed hyperparameter tuning.
    
    Args:
        estimator: ML estimator (untrained)
        param_grid: Parameter grid dictionary
        train_df: Training DataFrame
        evaluator: Regression evaluator
        num_folds: Number of CV folds
        parallelism: Parallel execution level
        seed: Random seed
        
    Returns:
        Tuple of (best model, tuning time in seconds)
    """
    # Build parameter grid
    builder = ParamGridBuilder()
    for param_name, param_values in param_grid.items():
        param = getattr(estimator, param_name)
        builder = builder.addGrid(param, param_values)
    
    param_maps = builder.build()
    
    # Create CrossValidator
    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_maps,
        evaluator=evaluator,
        numFolds=num_folds,
        parallelism=parallelism,
        seed=seed
    )
    
    # Fit with timing
    start_time = time.time()
    cv_model = cv.fit(train_df)
    tune_time = time.time() - start_time
    
    best_model = cv_model.bestModel
    
    return best_model, tune_time


def tune_model_with_tvs(
    estimator: Any,
    param_grid: Dict,
    train_df: DataFrame,
    evaluator: RegressionEvaluator,
    train_ratio: float = 0.8,
    parallelism: int = 4,
    seed: int = 42
) -> Tuple[Any, float]:
    """
    Tune model using TrainValidationSplit.
    
    Original implementation for faster hyperparameter tuning.
    
    Args:
        estimator: ML estimator (untrained)
        param_grid: Parameter grid dictionary
        train_df: Training DataFrame
        evaluator: Regression evaluator
        train_ratio: Training split ratio
        parallelism: Parallel execution level
        seed: Random seed
        
    Returns:
        Tuple of (best model, tuning time in seconds)
    """
    # Build parameter grid
    builder = ParamGridBuilder()
    for param_name, param_values in param_grid.items():
        param = getattr(estimator, param_name)
        builder = builder.addGrid(param, param_values)
    
    param_maps = builder.build()
    
    # Create TrainValidationSplit
    tvs = TrainValidationSplit(
        estimator=estimator,
        estimatorParamMaps=param_maps,
        evaluator=evaluator,
        trainRatio=train_ratio,
        parallelism=parallelism,
        seed=seed
    )
    
    # Fit with timing
    start_time = time.time()
    tvs_model = tvs.fit(train_df)
    tune_time = time.time() - start_time
    
    best_model = tvs_model.bestModel
    
    return best_model, tune_time


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default parameter grids for hyperparameter tuning.
    
    Original implementation with NYC Taxi-specific parameter ranges.
    
    Returns:
        Dictionary of parameter grids for each model type
    """
    param_grids = {
        'linear_regression': {
            'regParam': [0.0, 0.01, 0.1],
            'elasticNetParam': [0.0, 0.5, 1.0]
        },
        'decision_tree': {
            'maxDepth': [5, 10, 15],
            'maxBins': [32, 64],
            'minInstancesPerNode': [1, 5]
        },
        'random_forest': {
            'numTrees': [20, 50, 100],
            'maxDepth': [5, 10, 15],
            'maxBins': [32, 64]
        },
        'gbt': {
            'maxIter': [20, 50, 100],
            'maxDepth': [3, 5, 7],
            'stepSize': [0.01, 0.1, 0.3]
        },
        'glr': {
            'regParam': [0.0, 0.01, 0.1]
        }
    }
    
    return param_grids


def save_model(model: Any, path: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        path: Output path
    """
    model.save(path)
    print(f"Model saved to: {path}")


def load_model(path: str, model_type: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        path: Model path
        model_type: Type of model (lr, dt, rf, gbt, glr)
        
    Returns:
        Loaded model
    """
    model_classes = {
        'lr': LinearRegression,
        'dt': DecisionTreeRegressor,
        'rf': RandomForestRegressor,
        'gbt': GBTRegressor,
        'glr': GeneralizedLinearRegression
    }
    
    model_class = model_classes.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_class.load(path)
    return model
