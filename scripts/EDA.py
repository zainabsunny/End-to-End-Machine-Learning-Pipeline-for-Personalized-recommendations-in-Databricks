import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from collections import Counter

def perform_eda(combined_df, reviews_df):
    """
    Perform EDA on the combined and review datasets.
    Args:
    - combined_df: DataFrame containing combined structured and unstructured data.
    - reviews_df: DataFrame containing reviews with cleaned text.
    """

    # Summary statistics for numeric columns
    print("Summary Statistics for Numeric Columns:")
    combined_df.describe().show()

    # Specific statistics with null handling
    print("Specific Statistics for Selected Columns:")
    numeric_cols = ["cosmetic_price", "review_price", "product_rating", "product_rating_count"]
    for col_name in numeric_cols:
        combined_df = combined_df.withColumn(col_name, col(col_name).cast("double")).na.fill({col_name: 0})
    combined_df.select(*numeric_cols).describe().show()

    # Distribution of Prices
    print("Distribution of Prices:")
    cosmetic_price_data = combined_df.select("cosmetic_price").rdd.flatMap(lambda x: x).filter(lambda x: x is not None).collect()
    review_price_data = combined_df.select("review_price").rdd.flatMap(lambda x: x).filter(lambda x: x is not None).collect()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(cosmetic_price_data, bins=50, color="blue", alpha=0.7)
    plt.title("Distribution of Cosmetic Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(review_price_data, bins=50, color="purple", alpha=0.7)
    plt.title("Distribution of Review Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Distribution of Product Ratings
    print("Distribution of Product Ratings:")
    rating_data = combined_df.select("product_rating").rdd.flatMap(lambda x: x).filter(lambda x: x is not None).collect()
    plt.figure(figsize=(10, 6))
    sns.histplot(rating_data, bins=10, kde=True, color="green", alpha=0.6)
    plt.title("Distribution of Product Ratings")
    plt.xlabel("Product Rating")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Daily Customer Engagement Events
    print("Daily Customer Engagement Events:")
    combined_df = combined_df.withColumn("event_date", col("event_time").cast("date"))
    daily_engagement = combined_df.groupBy("event_date").count().orderBy("event_date")
    daily_engagement.show()

    date_data = daily_engagement.select("event_date").rdd.flatMap(lambda x: x).collect()
    engagement_data = daily_engagement.select("count").rdd.flatMap(lambda x: x).collect()

    plt.figure(figsize=(12, 6))
    plt.plot(date_data, engagement_data, color="blue", marker="o")
    plt.title("Daily Customer Engagement Events")
    plt.xlabel("Date")
    plt.ylabel("Event Count")
    plt.show()

    # Distribution of Event Types (Customer Engagement Levels)
    print("Distribution of Event Types:")
    event_counts = combined_df.groupBy("event_type_index").count().toPandas()
    plt.figure(figsize=(8, 5))
    plt.bar(event_counts["event_type_index"], event_counts["count"], color="skyblue")
    plt.title("Distribution of Event Types (Customer Engagement Levels)")
    plt.xlabel("Event Type (0=View, 1=Add-to-Cart, 2=Remove-from-Cart, 3=Purchase)")
    plt.ylabel("Count")
    plt.show()

    # Sentiment Distribution
    print("Sentiment Distribution in Reviews:")
    sentiment_counts = reviews_df.groupBy("review_label").count().toPandas()
    plt.figure(figsize=(8, 5))
    plt.bar(sentiment_counts["review_label"], sentiment_counts["count"], color="skyblue")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Sentiment vs. Product Rating
    print("Sentiment vs. Product Rating:")
    sentiment_rating = reviews_df.groupBy("review_label", "product_rating").count().toPandas()
    pivot_data = sentiment_rating.pivot(index="review_label", columns="product_rating", values="count")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="Blues", cbar=True)
    plt.title("Sentiment vs. Product Rating")
    plt.xlabel("Product Rating")
    plt.ylabel("Sentiment")
    plt.show()

    # Most Common Words in Reviews
    print("Most Common Words in Lemmatized Review Text:")
    lemmatized_texts = reviews_df.select("lemmatized_text").rdd.flatMap(lambda x: x).collect()
    all_words = " ".join([text for text in lemmatized_texts if text]).split()
    word_freq = Counter(all_words).most_common(20)
    words, counts = zip(*word_freq)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette="magma")
    plt.title("Top 20 Most Frequent Words in Lemmatized Review Text")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()

    # Top 10 Most Reviewed Products
    print("Top 10 Most Reviewed Products:")
    top_products = reviews_df.groupBy("product_title").count().orderBy(col("count").desc()).limit(10).toPandas()

    plt.figure(figsize=(12, 6))
    sns.barplot(x="count", y="product_title", data=top_products, palette="cool")
    plt.title("Top 10 Most Reviewed Products")
    plt.xlabel("Review Count")
    plt.ylabel("Product Title")
    plt.show()
    

    print("EDA Completed.")
