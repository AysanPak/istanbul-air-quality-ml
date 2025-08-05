# 03_pm25_forecasting.py
"""
Builds and evaluates a Random Forest Regressor model to forecast PM2.5 levels
using historical time-series data from Istanbul air quality monitoring stations.

The script follows a robust pipeline:
1.  Loads and prepares raw data from multiple CSV files.
2.  Engineers an extensive set of time-series features (lags, rolling stats, etc.).
3.  Cleans outliers and null values to ensure data quality.
4.  Builds a PySpark ML Pipeline to handle feature transformations.
5.  Performs a time-aware, station-specific train-test split (80/20).
6.  Trains a Random Forest model and evaluates its performance on the test set.
7.  Conducts an overfitting check by comparing train and test performance.
8.  Calculates and displays the most important features driving the predictions.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, year, lag, avg, stddev, hour, dayofweek,
    month, when, percent_rank
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def run_pm25_forecasting(spark, data_path):
    """
    Executes the entire PM2.5 forecasting pipeline from data loading to evaluation.

    Args:
        spark (SparkSession): The active Spark session.
        data_path (str): The path to the directory containing the raw input CSV files.
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: FORECASTING PM2.5 LEVELS WITH A RANDOM FOREST MODEL")
    print("="*80 + "\n")

    # === Step 1: Load and Prepare Raw Data ===
    print("--> Step 1/8: Loading and preparing raw data from CSV files...")
    df_raw = spark.read.option("header", "true").csv(data_path)

    # Automatically cast all pollutant columns to DoubleType
    cols_to_cast = [c for c in df_raw.columns if c not in ["station", "ReadTime"]]
    for column_name in cols_to_cast:
        df_raw = df_raw.withColumn(column_name, col(column_name).cast(DoubleType()))

    df_raw = df_raw.withColumn("ReadTime", to_timestamp("ReadTime"))
    df_filtered = df_raw.filter(year("ReadTime") >= 2020)
    
    print(f"   - Loaded {df_filtered.count()} rows of data from 2020 onwards.")
    print("--> Step 1/8: Complete.\n")


    # === Step 2: Feature Engineering ===
    print("--> Step 2/8: Engineering time-series features (lags, rolling stats)...")
    w = Window.partitionBy("station").orderBy("ReadTime")

    df_fe = (
        df_filtered
        .withColumn("lag_1h", lag("PM25", 1).over(w))
        .withColumn("lag_3h", lag("PM25", 3).over(w))
        .withColumn("lag_6h", lag("PM25", 6).over(w))
        .withColumn("lag_24h", lag("PM25", 24).over(w))
        .withColumn("rolling_mean_3h", avg("PM25").over(w.rowsBetween(-2, 0)))
        .withColumn("rolling_mean_6h", avg("PM25").over(w.rowsBetween(-5, 0)))
        .withColumn("rolling_mean_24h", avg("PM25").over(w.rowsBetween(-23, 0)))
        .withColumn("rolling_std_6h", stddev("PM25").over(w.rowsBetween(-5, 0)))
        .withColumn("rolling_std_24h", stddev("PM25").over(w.rowsBetween(-23, 0)))
        .withColumn("hour", hour("ReadTime"))
        .withColumn("day_of_week", dayofweek("ReadTime"))
        .withColumn("month", month("ReadTime"))
        .withColumn("is_weekend", when(dayofweek("ReadTime").isin([1, 7]), 1).otherwise(0))
        .withColumn("season", when(col("month").isin([12, 1, 2]), "winter")
                              .when(col("month").isin([3, 4, 5]), "spring")
                              .when(col("month").isin([6, 7, 8]), "summer")
                              .otherwise("fall"))
        .withColumn("diff_1h", col("PM25") - lag("PM25", 1).over(w))
        .withColumn("diff_24h", col("PM25") - lag("PM25", 24).over(w))
        .withColumn("pm25_to_pm10_ratio", when(col("PM10") > 0, col("PM25") / col("PM10")).otherwise(0))
    )
    print("--> Step 2/8: Complete.\n")


    # === Step 3: Outlier and Null Value Cleaning ===
    print("--> Step 3/8: Removing outliers and null values...")
    stats = df_fe.select(avg("PM25").alias("mean"), stddev("PM25").alias("std")).collect()[0]
    df_fe = df_fe.withColumn("zscore", (col("PM25") - stats["mean"]) / stats["std"])
    df_fe = df_fe.filter(col("zscore").between(-4, 4)).drop("zscore")

    # Define feature columns and drop any rows with nulls in them
    feature_cols = [c for c in df_fe.columns if c not in ["station", "ReadTime", "PM25", "season", "zscore"]]
    df_model = df_fe.dropna(subset=["PM25"] + feature_cols)
    print("--> Step 3/8: Complete.\n")


    # === Step 4: Define ML Pipeline for Feature Transformation ===
    print("--> Step 4/8: Defining machine learning pipeline...")
    label_col = "PM25"
    
    indexer = StringIndexer(inputCol="season", outputCol="season_index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="season_index", outputCol="season_ohe", dropLast=False)
    
    final_feature_cols = feature_cols + ["season_ohe"]
    assembler = VectorAssembler(inputCols=final_feature_cols, outputCol="features")

    pipeline = Pipeline(stages=[indexer, encoder, assembler])
    print("--> Step 4/8: Complete.\n")


    # === Step 5: Fit Pipeline and Perform Time-Based Train-Test Split ===
    print("--> Step 5/8: Fitting pipeline and splitting data (80% train / 20% test)...")
    pipeline_model = pipeline.fit(df_model)
    df_ml = pipeline_model.transform(df_model)

    w_station = Window.partitionBy("station").orderBy("ReadTime")
    df_ml = df_ml.withColumn("rank", percent_rank().over(w_station))

    train_df = df_ml.filter(col("rank") <= 0.8).drop("rank")
    test_df  = df_ml.filter(col("rank") > 0.8).drop("rank")
    
    print(f"   - Training set size: {train_df.count()}, Test set size: {test_df.count()}")
    print("--> Step 5/8: Complete.\n")


    # === Step 6: Train Random Forest Model ===
    print("--> Step 6/8: Training Random Forest model...")
    rf = RandomForestRegressor(
        labelCol=label_col,
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    model = rf.fit(train_df)
    print("--> Step 6/8: Complete.\n")


    # === Step 7: Evaluate Model Performance and Check for Overfitting ===
    print("--> Step 7/8: Evaluating model performance...")
    train_pred = model.transform(train_df)
    test_pred = model.transform(test_df)

    metrics = ["rmse", "mae", "r2"]
    results = {}
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName=metric)
        results[f"train_{metric}"] = evaluator.evaluate(train_pred)
        results[f"test_{metric}"] = evaluator.evaluate(test_pred)

    print("\n--- MODEL PERFORMANCE METRICS ---\n")
    print(f"{'Metric':<10} | {'Train Set':<15} | {'Test Set':<15}")
    print("-" * 45)
    print(f"{'RMSE':<10} | {results['train_rmse']:<15.3f} | {results['test_rmse']:<15.3f}")
    print(f"{'MAE':<10} | {results['train_mae']:<15.3f} | {results['test_mae']:<15.3f}")
    print(f"{'R-squared':<10} | {results['train_r2']:<15.4f} | {results['test_r2']:<15.4f}")
    print("\n--> Step 7/8: Complete.\n")


    # === Step 8: Analyze Feature Importances ===
    print("--> Step 8/8: Analyzing feature importances...")
    importances = model.featureImportances.toArray()
    feature_names = pipeline_model.stages[-1].getInputCols()
    
    importance_df = spark.createDataFrame(
        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True),
        ["feature", "importance"]
    )
    
    print("\n--- TOP 15 MOST IMPORTANT FEATURES ---\n")
    importance_df.show(15, truncate=False)
    print("--> Step 8/8: Complete. Forecasting analysis finished.\n")


def main():
    """Main function to initialize Spark and run the forecasting pipeline."""
    spark = SparkSession.builder \
        .appName("IstanbulPM25Forecasting") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # The path to the directory containing all raw CSV files.
    # Spark will read all .csv files within this folder.
    DATA_PATH = "data/raw_air_quality_data/"

    # Run the full forecasting pipeline
    run_pm25_forecasting(spark, DATA_PATH)

    spark.stop()


if __name__ == "__main__":
    main()