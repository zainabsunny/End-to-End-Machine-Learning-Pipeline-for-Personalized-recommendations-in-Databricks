from pyspark.sql.functions import col, datediff, current_date, countDistinct
from pyspark.ml.feature import StringIndexer

# -----------------------------
# 1) Customer Engagement (Outcome Variable Y)
# -----------------------------

def add_customer_engagement(cosmetic_df):
    """
    Encode event_type to map engagement levels.
    """
    print("Running add_customer_engagement...")
    if "event_type_index" in cosmetic_df.columns:
        cosmetic_df = cosmetic_df.drop("event_type_index")
    indexer = StringIndexer(inputCol="event_type", outputCol="event_type_index")
    cosmetic_df = indexer.fit(cosmetic_df).transform(cosmetic_df)
    return cosmetic_df

# -----------------------------
# 2) Predictor Variables (X)
# -----------------------------

def add_predictor_features(cosmetic_df):
    """
    Add structured predictor features to the DataFrame.
    """
    # Recency
    cosmetic_df = cosmetic_df.withColumn("recency", datediff(current_date(), col("event_time")))

    # Frequency
    frequency_df = cosmetic_df.groupBy("user_session").count().withColumnRenamed("count", "frequency")
    cosmetic_df = cosmetic_df.join(frequency_df, on="user_session", how="left")

    # Product Popularity
    product_popularity = (
        cosmetic_df.filter(col("event_type") == "purchase")
        .groupBy("cosmetic_product_id")
        .count()
        .withColumnRenamed("count", "popularity")
    )
    cosmetic_df = cosmetic_df.join(product_popularity, on="cosmetic_product_id", how="left")

    # Session Diversity
    session_diversity = (
        cosmetic_df.groupBy("user_session")
        .agg(countDistinct("category_code").alias("session_diversity"))
    )
    cosmetic_df = cosmetic_df.join(session_diversity, on="user_session", how="left")

    return cosmetic_df

# -----------------------------
# 3) Process Reviews
# -----------------------------

def process_reviews_df(reviews_df):
    """Clean and process reviews DataFrame by handling missing values."""

    # Print available columns for debugging
    print(f"✅ Available columns in reviews_df: {reviews_df.columns}")

    # Use correct column names based on available data
    reviews_df = reviews_df.fillna({
        "review_title": "No title",
        "review_text": "No review"
    })

    return reviews_df

