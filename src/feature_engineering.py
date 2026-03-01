"""
Feature Engineering Module
===========================

NYC Taxi-specific feature engineering utilities.
All implementations are original, written from PySpark documentation.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, dayofweek, month, year, when, unix_timestamp,
    avg, count
)
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from typing import List


def create_time_features(df: DataFrame, config: dict) -> DataFrame:
    """
    Create time-based features from pickup datetime.
    
    Original implementation with NYC-specific rush hour logic.
    
    Args:
        df: Input DataFrame with tpep_pickup_datetime
        config: Configuration dictionary with rush hour definitions
        
    Returns:
        DataFrame with time-based features
    """
    features_config = config['project']['features']
    
    # Extract time components
    df_time = df \
        .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
        .withColumn("pickup_dow", dayofweek(col("tpep_pickup_datetime"))) \
        .withColumn("pickup_month", month(col("tpep_pickup_datetime"))) \
        .withColumn("pickup_year", year(col("tpep_pickup_datetime")))
    
    # Weekend indicator
    df_time = df_time.withColumn(
        "is_weekend",
        when(col("pickup_dow").isin([1, 7]), 1).otherwise(0)
    )
    
    # Rush hour indicator (NYC-specific: 7-9am, 5-7pm)
    morning_start = features_config['rush_hour_morning_start']
    morning_end = features_config['rush_hour_morning_end']
    evening_start = features_config['rush_hour_evening_start']
    evening_end = features_config['rush_hour_evening_end']
    
    df_time = df_time.withColumn(
        "is_rush_hour",
        when(
            ((col("pickup_hour") >= morning_start) & (col("pickup_hour") < morning_end)) |
            ((col("pickup_hour") >= evening_start) & (col("pickup_hour") < evening_end)),
            1
        ).otherwise(0)
    )
    
    # Late night indicator (10pm - 5am)
    late_start = features_config['late_night_start']
    late_end = features_config['late_night_end']
    
    df_time = df_time.withColumn(
        "is_late_night",
        when(
            (col("pickup_hour") >= late_start) | (col("pickup_hour") < late_end),
            1
        ).otherwise(0)
    )
    
    return df_time


def create_trip_features(df: DataFrame, config: dict) -> DataFrame:
    """
    Create trip-based features.
    
    Original implementation for NYC Taxi trip metrics.
    
    Args:
        df: Input DataFrame with pickup/dropoff times and distance
        config: Configuration dictionary
        
    Returns:
        DataFrame with trip-based features
    """
    features_config = config['project']['features']
    
    # Trip duration in seconds
    df_trip = df.withColumn(
        "trip_duration_seconds",
        unix_timestamp(col("tpep_dropoff_datetime")) -
        unix_timestamp(col("tpep_pickup_datetime"))
    )
    
    # Trip duration in minutes
    df_trip = df_trip.withColumn(
        "trip_duration_minutes",
        col("trip_duration_seconds") / 60.0
    )
    
    # Average speed (mph)
    df_trip = df_trip.withColumn(
        "speed_mph",
        when(
            col("trip_duration_seconds") > 0,
            (col("trip_distance") / col("trip_duration_seconds")) * 3600.0
        ).otherwise(0.0)
    )
    
    # Fare per mile
    df_trip = df_trip.withColumn(
        "fare_per_mile",
        when(
            col("trip_distance") > 0,
            col("fare_amount") / col("trip_distance")
        ).otherwise(0.0)
    )
    
    # Tip percentage
    df_trip = df_trip.withColumn(
        "tip_percentage",
        when(
            col("fare_amount") > 0,
            (col("tip_amount") / col("fare_amount")) * 100.0
        ).otherwise(0.0)
    )
    
    # Filter unrealistic speeds (NYC traffic constraint)
    max_speed = features_config['max_speed_mph']
    df_trip = df_trip.filter(col("speed_mph") <= max_speed)
    
    return df_trip


def create_location_features(df: DataFrame, config: dict) -> DataFrame:
    """
    Create location-based features for NYC geography.
    
    Original implementation with NYC-specific zone logic.
    
    Args:
        df: Input DataFrame with PULocationID and DOLocationID
        config: Configuration dictionary with zone definitions
        
    Returns:
        DataFrame with location-based features
    """
    features_config = config['project']['features']
    
    # Airport indicators (JFK, LGA, EWR)
    jfk_zones = features_config['jfk_zones']
    lga_zones = features_config['lga_zones']
    ewr_zones = features_config.get('ewr_zones', [])
    
    airport_zones = list(set(jfk_zones + lga_zones + ewr_zones))
    
    df_location = df \
        .withColumn(
            "is_airport_pickup",
            when(col("PULocationID").isin(airport_zones), 1).otherwise(0)
        ) \
        .withColumn(
            "is_airport_dropoff",
            when(col("DOLocationID").isin(airport_zones), 1).otherwise(0)
        )
    
    # Manhattan indicators (approximate zone range)
    manhattan_min = features_config['manhattan_zone_min']
    manhattan_max = features_config['manhattan_zone_max']
    
    df_location = df_location \
        .withColumn(
            "is_manhattan_pickup",
            when(
                (col("PULocationID") >= manhattan_min) &
                (col("PULocationID") <= manhattan_max),
                1
            ).otherwise(0)
        ) \
        .withColumn(
            "is_manhattan_dropoff",
            when(
                (col("DOLocationID") >= manhattan_min) &
                (col("DOLocationID") <= manhattan_max),
                1
            ).otherwise(0)
        )
    
    # Cross-borough trip indicator
    df_location = df_location.withColumn(
        "is_cross_borough",
        when(
            (col("is_manhattan_pickup") != col("is_manhattan_dropoff")),
            1
        ).otherwise(0)
    )
    
    return df_location


def create_aggregated_features(df: DataFrame) -> DataFrame:
    """
    Create aggregated features based on vendor and time.
    
    Original implementation for behavioral features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with aggregated features
    """
    # Vendor-level aggregations
    vendor_stats = df.groupBy("VendorID").agg(
        avg("fare_amount").alias("vendor_avg_fare"),
        avg("trip_distance").alias("vendor_avg_distance"),
        avg("tip_amount").alias("vendor_avg_tip")
    )
    
    df_agg = df.join(vendor_stats, "VendorID", "left")
    
    # Hour-level aggregations
    hour_stats = df.groupBy("pickup_hour").agg(
        avg("fare_amount").alias("hour_avg_fare"),
        avg("trip_distance").alias("hour_avg_distance"),
        count("*").alias("hour_trip_count")
    )
    
    df_agg = df_agg.join(hour_stats, "pickup_hour", "left")
    
    # Payment type aggregations
    payment_stats = df.groupBy("payment_type").agg(
        avg("tip_percentage").alias("payment_avg_tip_pct")
    )
    
    df_agg = df_agg.join(payment_stats, "payment_type", "left")
    
    return df_agg


def build_feature_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    target_col: str = "fare_amount"
) -> Pipeline:
    """
    Build complete feature engineering pipeline.
    
    Original implementation combining encoding, assembly, and scaling.
    
    Args:
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        target_col: Target variable name
        
    Returns:
        PySpark ML Pipeline
    """
    stages = []
    
    # Categorical encoding
    if categorical_cols:
        # String indexing
        indexers = [
            StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_index",
                handleInvalid="keep"
            )
            for col_name in categorical_cols
        ]
        stages.extend(indexers)
        
        # One-hot encoding
        encoder = OneHotEncoder(
            inputCols=[f"{col}_index" for col in categorical_cols],
            outputCols=[f"{col}_encoded" for col in categorical_cols],
            handleInvalid="keep",
            dropLast=True
        )
        stages.append(encoder)
    
    # Feature assembly
    feature_cols = numeric_cols.copy()
    if categorical_cols:
        feature_cols.extend([f"{col}_encoded" for col in categorical_cols])
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    stages.append(assembler)
    
    # Feature scaling
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    stages.append(scaler)
    
    # Create pipeline
    pipeline = Pipeline(stages=stages)
    
    return pipeline


def get_feature_names(
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> List[str]:
    """
    Get list of feature names in order.
    
    Args:
        categorical_cols: Categorical column names
        numeric_cols: Numeric column names
        
    Returns:
        Ordered list of feature names
    """
    feature_names = numeric_cols.copy()
    
    # Add categorical feature names (after one-hot encoding)
    for col_name in categorical_cols:
        feature_names.append(f"{col_name}_encoded")
    
    return feature_names


def select_final_features(
    df: DataFrame,
    feature_cols: List[str],
    target_col: str = "fare_amount",
    id_cols: List[str] = None
) -> DataFrame:
    """
    Select final columns for model training.
    
    Args:
        df: Input DataFrame
        feature_cols: Feature column names
        target_col: Target variable name
        id_cols: ID columns to keep (optional)
        
    Returns:
        DataFrame with selected columns
    """
    select_cols = ["features", target_col]
    
    if id_cols:
        select_cols = id_cols + select_cols
    
    # Add some useful columns for analysis
    analysis_cols = [
        "trip_distance",
        "passenger_count",
        "is_rush_hour",
        "is_weekend",
        "pickup_hour"
    ]
    
    for col_name in analysis_cols:
        if col_name in df.columns and col_name not in select_cols:
            select_cols.append(col_name)
    
    df_final = df.select(*select_cols)
    
    return df_final
