from pyspark.sql import functions as F
from pyspark.sql.window import Window

def evaluate_precision_recall(recommendations_df, ground_truth_df, k=10):
    """
    Evaluates Precision@K and Recall@K for recommendations.

    Args:
        recommendations_df (DataFrame): DataFrame with columns ['user_session', 'recommended_product_id', 'rank'].
        ground_truth_df (DataFrame): DataFrame with columns ['user_session', 'cosmetic_product_id'] (actual purchases).
        k (int): The number of top recommendations to consider.

    Returns:
        tuple: Precision@K and Recall@K values.
    """
    # Limit recommendations to top-K
    recommendations_top_k = recommendations_df.filter(F.col("rank") <= k)

    # Join recommendations with ground truth to find relevant recommendations
    relevant_recommendations = recommendations_top_k.join(
        ground_truth_df,
        (recommendations_top_k["user_session"] == ground_truth_df["user_session"]) &
        (recommendations_top_k["recommended_product_id"] == ground_truth_df["cosmetic_product_id"]),
        how="inner"
    )

    # Calculate precision@K
    total_recommendations = recommendations_top_k.groupBy("user_session").count().withColumnRenamed("count", "total_recommendations")
    relevant_count = relevant_recommendations.groupBy("user_session").count().withColumnRenamed("count", "relevant_recommendations")
    precision_df = total_recommendations.join(relevant_count, on="user_session", how="left").fillna(0)
    precision_df = precision_df.withColumn("precision", F.col("relevant_recommendations") / F.col("total_recommendations"))
    precision_at_k = precision_df.select(F.mean("precision")).collect()[0][0]

    # Calculate recall@K
    total_relevant = ground_truth_df.groupBy("user_session").count().withColumnRenamed("count", "total_relevant")
    recall_df = total_relevant.join(relevant_count, on="user_session", how="left").fillna(0)
    recall_df = recall_df.withColumn("recall", F.col("relevant_recommendations") / F.col("total_relevant"))
    recall_at_k = recall_df.select(F.mean("recall")).collect()[0][0]

    return precision_at_k, recall_at_k
