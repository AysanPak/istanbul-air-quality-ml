# 01_district_clustering.py
"""
Performs K-Means clustering on Istanbul air quality data to group districts
with similar pollution patterns, identifying distinct pollution zones.

This script directly mirrors the logic from the project's primary analysis file.

Steps:
1.  Loads the merged Istanbul air quality data from a Parquet file.
2.  Filters for key pollutants and applies a unit correction for CO values.
3.  Pivots the data to create a feature profile (average pollution) for each district.
4.  Applies winsorization (capping at the 95th percentile) to handle outliers.
5.  Uses a StandardScaler and VectorAssembler to prepare data for clustering.
6.  Tests multiple values of K (3, 4, 5) and selects the best one based on the
    highest silhouette score.
7.  Fits the final K-Means model and prints the districts belonging to each of the
    four identified clusters, along with their dominant pollutant.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark import StorageLevel

def run_district_clustering(spark, df_all_data):
    """
    Executes the K-Means clustering analysis to group districts.

    Args:
        spark (SparkSession): The active Spark session.
        df_all_data (DataFrame): The input DataFrame loaded from the Parquet file.
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: IDENTIFYING DISTRICT POLLUTION PATTERNS VIA K-MEANS CLUSTERING")
    print("="*80 + "\n")

    # === Step 1: Create District-Level Pollution Profiles ===
    print("--> Step 1/5: Creating district pollution profiles...")
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

    df_clean = df_all_data.filter(col("parameter").isin(pollutants)) \
                          .filter(col("value") > 0) \
                          .filter(col("value") < 1000)

    # Apply CO unit correction (values appear to be 1000x too large)
    print("   - Applying CO unit correction...")
    df_clean = df_clean.withColumn("value",
                                   when(col("parameter") == "co", col("value") / 1000)
                                   .otherwise(col("value")))

    # Pivot to create features for each district based on average pollutant values
    district_features_pivoted = df_clean.groupBy("district") \
                                        .pivot("parameter", pollutants) \
                                        .agg(expr("avg(value)"))

    # Fill nulls and rename columns to `_avg` as required by the assembler
    district_features = district_features_pivoted
    for pollutant in pollutants:
        district_features = district_features.fillna(0, subset=[pollutant]) \
                                           .withColumnRenamed(pollutant, f"{pollutant}_avg")
    print("--> Step 1/5: Complete.\n")


    # === Step 2: Handle Outliers with Winsorization ===
    print("--> Step 2/5: Capping extreme values at the 95th percentile...")
    feature_cols = [f"{p}_avg" for p in pollutants]

    for col_name in feature_cols:
        percentile_95 = district_features.select(
            expr(f"percentile_approx({col_name}, 0.95)").alias("p95")
        ).collect()[0]['p95']

        district_features = district_features.withColumn(col_name,
            when(col(col_name) > percentile_95, percentile_95)
            .otherwise(col(col_name))
        )
    print("--> Step 2/5: Complete. Outlier capping finished.\n")

    district_features = district_features.filter(col("district").isNotNull()) \
                                       .persist(StorageLevel.MEMORY_AND_DISK)


    # === Step 3: Prepare Features for Clustering ===
    print("--> Step 3/5: Assembling and scaling feature vectors...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    print("--> Step 3/5: Complete.\n")


    # === Step 4: Determine the Optimal Number of Clusters (K) ===
    print("--> Step 4/5: Evaluating different values of K to find the best fit...")
    results = []
    for k in [3, 4, 5]:
        kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features", predictionCol="cluster")
        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = pipeline.fit(district_features)
        clustered = model.transform(district_features)

        evaluator = ClusteringEvaluator(featuresCol="scaled_features", predictionCol="cluster", metricName="silhouette")
        silhouette = evaluator.evaluate(clustered)
        results.append((k, silhouette))
        print(f"  - For K={k}, Silhouette Score = {silhouette:.3f}")

    best_k = max(results, key=lambda x: x[1])[0]
    best_silhouette = max(results, key=lambda x: x[1])[1]
    print(f"\nOptimal K selected: {best_k} (Silhouette Score: {best_silhouette:.3f})\n")
    print("--> Step 4/5: Complete.\n")


    # === Step 5: Final Clustering and Results ===
    print("--> Step 5/5: Running final clustering and displaying results...")
    kmeans_final = KMeans(k=best_k, seed=42, featuresCol="scaled_features", predictionCol="cluster")
    pipeline_final = Pipeline(stages=[assembler, scaler, kmeans_final])
    model_final = pipeline_final.fit(district_features)
    clustered_final = model_final.transform(district_features)

    print(f"\n--- ISTANBUL DISTRICT POLLUTION CLUSTERS (K={best_k}) ---\n")
    for cluster_id in range(best_k):
        cluster_df = clustered_final.filter(col("cluster") == cluster_id)
        cluster_districts = cluster_df.select("district").collect()
        district_names = [row['district'] for row in cluster_districts]

        # Calculate cluster characteristics to find dominant pollutant
        cluster_stats = cluster_df.select(
            *[expr(f"avg({p}_avg)").alias(f"avg_{p}") for p in pollutants]
        ).collect()[0]

        avg_pollutants = {p: cluster_stats[f"avg_{p}"] for p in pollutants}
        dominant_pollutant_name = max(avg_pollutants, key=avg_pollutants.get)
        dominant_pollutant_value = avg_pollutants[dominant_pollutant_name]
        co_value = avg_pollutants['co']

        print(f"--- Cluster {cluster_id} ({len(district_names)} districts) ---")
        print(f"  Districts: {', '.join(sorted(district_names))}")
        print(f"  Dominant Pollutant: {dominant_pollutant_name.upper()} ({dominant_pollutant_value:.1f} µg/m³)")
        print(f"  Average CO Level: {co_value:.1f} mg/m³ (corrected units)\n")

    district_features.unpersist()
    print("--> Step 5/5: Complete. Analysis finished.\n")


def main():
    """Main function to initialize Spark and run the clustering analysis."""
    spark = SparkSession.builder \
        .appName("IstanbulDistrictClustering") \
        .master("local[*]") \
        .getOrCreate()

    # The path to the Parquet file within repository's data folder
    DATA_PATH = "data/processed_data/istanbul_all_stations_merged.parquet"

    print(f"Loading data from: {DATA_PATH}\n")
    df_all_data = spark.read.parquet(DATA_PATH)
    
    # Cache the DataFrame for reuse in other analyses
    df_all_data.persist(StorageLevel.MEMORY_AND_DISK)
    
    print(f"Data loaded successfully. Total rows: {df_all_data.count()}")
    print(f"Schema: {df_all_data.columns}\n")

    # Run the clustering analysis
    run_district_clustering(spark, df_all_data)

    df_all_data.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()