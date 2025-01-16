from pyspark.sql.functions import udf, col, datediff, current_date, min, max, countDistinct
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.feature import StringIndexer
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1) Unstructured Data: Text Embeddings (no spaCy)
# -----------------------------

def get_stella_embeddings(text):
    """
    Generate embeddings for the given text using dunzhang/stella_en_400M_v5.
    Returns a list of floats (the embedding vector).
    """
    try:
        if not text or not isinstance(text, str):
            return []
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        # Forward pass
        outputs = model(**inputs)
        # Average the last hidden states to get a single embedding vector
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()[0]
        return embedding
    except Exception as e:
        print(f"Error generating embeddings for text: {text}. Error: {e}")
        return []

# We'll define our model/tokenizer at a higher scope so we only load them once
tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")
model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5")

# Create a UDF using the above embedding function
stella_embedding_udf = udf(get_stella_embeddings, ArrayType(FloatType()))

def process_reviews_df(reviews_df):
    """
    Clean and process reviews DataFrame using stella_en_400M_v5 embeddings
    """
    # Fill NA in text columns
    reviews_df = reviews_df.fillna({
        "review_title_clean": "No title",
        "review_text_clean": "No review"
    })
    
    # Generate embeddings for the review title and text
    reviews_df = reviews_df.withColumn("review_title_embedding", stella_embedding_udf(col("review_title_clean")))
    reviews_df = reviews_df.withColumn("review_text_embedding", stella_embedding_udf(col("review_text_clean")))

    return reviews_df

# -----------------------------
# 2) Structured Data: Outcome Variable (Y) and Predictor Variables (X)
# -----------------------------

def add_customer_engagement(cosmetic_df): 
    """Encode event_type to map engagement levels."""
    print("Updated add_customer_engagement function is running...")
    if "event_type_index" in cosmetic_df.columns:
        cosmetic_df = cosmetic_df.drop("event_type_index")
    indexer = StringIndexer(inputCol="event_type", outputCol="event_type_index")
    cosmetic_df = indexer.fit(cosmetic_df).transform(cosmetic_df)
    return cosmetic_df

def add_predictor_features(cosmetic_df):
    """Add structured predictor features to the DataFrame."""
    # Recency
    cosmetic_df = cosmetic_df.withColumn("recency", datediff(current_date(), col("event_time")))

    # Frequency
    frequency_df = (
        cosmetic_df.groupBy("user_session")
        .count()
        .withColumnRenamed("count", "frequency")
    )
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
