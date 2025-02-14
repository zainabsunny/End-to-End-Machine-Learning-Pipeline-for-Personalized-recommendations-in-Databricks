from transformers import pipeline
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

# ---------------------------------------
# 1) Load lightweight transformers for topic classification and sentiment analysis 
# ---------------------------------------

topic_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ---------------------------------------
# 2) Topic categories  
# ---------------------------------------

topic_labels = ["product quality", "packaging", "price", "customer service", "delivery", "ingredients", "effectiveness"]


# UDF for topic extraction
def extract_topics(review_text):
    if not review_text:
        return []
    result = topic_pipeline(review_text, topic_labels)
    topics = [label for label, score in zip(result['labels'], result['scores']) if score > 0.4]  # Adjust threshold
    return topics

# UDF for sentiment analysis
def get_sentiment(review_text):
    if not review_text:
        return []
    sentiments = sentiment_pipeline(review_text)
    return [(sent['label'], sent['score']) for sent in sentiments]

# Register PySpark UDFs
topic_udf = udf(extract_topics, ArrayType(StringType()))
sentiment_udf = udf(get_sentiment, ArrayType(StructType([
    StructField("label", StringType(), True),
    StructField("score", StringType(), True)
])))

# ---------------------------------------
# 3) LLM processing 
# ---------------------------------------

def process_reviews(spark_df):
    """
    Process the reviews DataFrame by extracting topics and sentiments.

    Args:
        spark_df (pyspark.sql.DataFrame): Input DataFrame containing review text.

    Returns:
        pyspark.sql.DataFrame: Processed DataFrame with topics and sentiments.
    """
    df_processed = (
        spark_df
        .withColumn("topics", topic_udf(col("review_text")))
        .withColumn("sentiment", sentiment_udf(col("review_text")))
    )
    return df_processed

# ---------------------------------------
# 3) Analyze topic distribution
# ---------------------------------------

def analyze_topics(spark_df):
    """
    Analyzes topic distribution by event type.

    Args:
        spark_df (pyspark.sql.DataFrame): Processed DataFrame with topics.

    Returns:
        pyspark.sql.DataFrame: Topic engagement DataFrame.
    """
    df_topics = spark_df.withColumn("topic", explode(col("topics")))
    df_topic_count = df_topics.groupBy("topic", "event_type").count().orderBy("count", ascending=False)
    return df_topic_count
