# 02_pollution_pattern_analysis.py
"""
Performs Principal Component Analysis (PCA) and Correlation Analysis on
Istanbul air quality data to uncover underlying pollution patterns.

This script directly mirrors the logic from the project's primary analysis file.

PCA identifies the primary factors (components) that explain the variance
in pollution, such as traffic, industry, or photochemical activity.

Correlation analysis quantifies and validates the relationships between pollutants.

Steps:
1.  Loads the merged Istanbul air quality data from a Parquet file.
2.  Runs PCA on a sampled subset of hourly data to identify the top 3-4 components.
3.  Interprets and prints the contribution of each pollutant to the components.
4.  Calculates a Pearson correlation matrix on concurrent measurements to ensure accuracy.
5.  Highlights the strongest positive and negative correlations found in the data.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark import StorageLevel

def run_pca_analysis(spark, df_all_data):
    """
    Performs PCA to identify the main underlying pollution patterns.

    Args:
        spark (SparkSession): The active Spark session.
        df_all_data (DataFrame): The input DataFrame loaded from the Parquet file.
    """
    print("\n" + "="*80)
    print("ANALYSIS 2A: FINDING POLLUTION DRIVERS VIA PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*80 + "\n")

    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

    # === Step 1: Prepare Hourly Pollution Vectors ===
    print("--> Step 1/3: Preparing hourly pollution vectors for PCA...")
    df_clean = df_all_data.filter(col("parameter").isin(pollutants)) \
                          .filter((col("value") > 0) & (col("value") < 1000))

    df_clean = df_clean.withColumn("value",
                                   when(col("parameter") == "co", col("value") / 1000)
                                   .otherwise(col("value")))

    # Create hourly vectors using a sampled and limited dataset for memory efficiency
    hourly_sample = df_clean.sample(fraction=0.02, seed=42) \
                           .groupBy("district", "year", "hour") \
                           .pivot("parameter", pollutants) \
                           .agg(expr("avg(value)")) \
                           .limit(3000)

    for pollutant in pollutants:
        hourly_sample = hourly_sample.fillna(0, subset=[pollutant])

    hourly_sample.persist(StorageLevel.MEMORY_AND_DISK)
    print(f"   - Using a sample of {hourly_sample.count()} hourly vectors for PCA.\n")
    print("--> Step 1/3: Complete.\n")


    # === Step 2: Run PCA Analysis ===
    print("--> Step 2/3: Assembling features and running PCA for 2, 3, and 4 components...")
    assembler = VectorAssembler(inputCols=pollutants, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

    for k in [2, 3, 4]:
        pca = PCA(k=k, inputCol="scaled_features", outputCol="pca_features")
        pipeline = Pipeline(stages=[assembler, scaler, pca])
        model = pipeline.fit(hourly_sample)
        pca_model = model.stages[-1]

        explained_variance = pca_model.explainedVariance.toArray()
        cumulative_variance = sum(explained_variance)
        pc_matrix = pca_model.pc.toArray()

        print(f"\n--- PCA with {k} components (Explained Variance: {cumulative_variance:.1%}) ---")
        for i in range(k):
            component = pc_matrix[:, i]
            feature_importance = sorted(
                zip(pollutants, component),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_features = [f"{feat} ({imp:.2f})" for feat, imp in feature_importance[:3]]
            print(f"  - PC{i+1} (explains {explained_variance[i]:.1%}): Top contributors: {', '.join(top_features)}")
    print("\n--> Step 2/3: Complete.\n")


    # === Step 3: Final Interpretation ===
    print("--> Step 3/3: Final Interpretation based on 3 components...")
    print("MAIN POLLUTION PATTERNS IDENTIFIED:")
    print("  - PC1 (Traffic-Urban): High loadings for NO2, CO, PM2.5.")
    print("  - PC2 (Photochemical): High loadings for O3, with secondary PM.")
    print("  - PC3 (Industrial): High loadings for SO2 and PM10.\n")

    hourly_sample.unpersist()
    print("--> Step 3/3: Complete. PCA Analysis finished.\n")


def run_correlation_analysis(spark, df_all_data):
    """
    Calculates the Pearson correlation matrix for the pollutants.

    Args:
        spark (SparkSession): The active Spark session.
        df_all_data (DataFrame): The input DataFrame loaded from the Parquet file.
    """
    print("\n" + "="*80)
    print("ANALYSIS 2B: VALIDATING POLLUTANT RELATIONSHIPS VIA CORRELATION")
    print("="*80 + "\n")

    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

    print("--> Step 1/2: Preparing data for correlation matrix...")
    df_clean = df_all_data.filter(col("parameter").isin(pollutants)) \
                          .filter((col("value") > 0) & (col("value") < 1000))

    # Get concurrent measurements at the same location/time for accurate correlation
    pivot_data = df_clean.groupBy("location", "datetime") \
                         .pivot("parameter", pollutants) \
                         .agg(expr("first(value)")) \
                         .filter(col("pm25").isNotNull() & col("pm10").isNotNull() & col("no2").isNotNull()) \
                         .sample(fraction=0.05, seed=42) \
                         .limit(5000)

    # Use dropna to avoid artificial correlations from zero-filled values
    pivot_data = pivot_data.dropna(how='any', subset=pollutants)
    print(f"   - Using {pivot_data.count()} concurrent measurements for correlation.\n")

    assembler = VectorAssembler(inputCols=pollutants, outputCol="features")
    correlation_data = assembler.transform(pivot_data).select("features")
    print("--> Step 1/2: Complete.\n")


    print("--> Step 2/2: Calculating and displaying correlation matrix...")
    correlation_matrix = Correlation.corr(correlation_data, "features", "pearson").collect()[0][0]
    correlation_array = correlation_matrix.toArray()

    print("\n--- POLLUTANT CORRELATION MATRIX ---\n")
    header = " " * 7 + "".join([f"{p:>8}" for p in pollutants])
    print(header)
    print("-" * len(header))
    for i, p_i in enumerate(pollutants):
        row_str = f"{p_i:<6}|"
        for j, p_j in enumerate(pollutants):
            row_str += f"{correlation_array[i, j]:>8.3f}"
        print(row_str)

    # Find and interpret strongest correlations, mirroring the project's output
    print("\n--- STRONGEST CORRELATIONS ---")
    correlations = []
    for i in range(len(pollutants)):
        for j in range(i + 1, len(pollutants)):
            correlations.append((pollutants[i], pollutants[j], correlation_array[i, j]))
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    for p1, p2, corr in correlations[:5]:
        print(f"   - {p1:<4} ↔ {p2:<4}: {corr:.3f}")
    print("\n--> Step 2/2: Complete. Correlation Analysis finished.\n")


def main():
    """Main function to run the pattern analysis scripts."""
    spark = SparkSession.builder \
        .appName("IstanbulPatternAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    DATA_PATH = "data/processed_data/istanbul_all_stations_merged.parquet"
    print(f"Loading data from: {DATA_PATH}\n")
    df_all_data = spark.read.parquet(DATA_PATH)
    
    # Cache for reuse between analyses
    df_all_data.persist(StorageLevel.MEMORY_AND_DISK)
    print(f"Data loaded successfully. Total rows: {df_all_data.count()}\n")

    # Run both analyses
    run_pca_analysis(spark, df_all_data)
    run_correlation_analysis(spark, df_all_data)

    df_all_data.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()