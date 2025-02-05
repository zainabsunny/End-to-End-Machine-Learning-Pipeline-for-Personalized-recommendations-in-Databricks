from pyspark.sql.functions import udf, col, explode, array, lit, expr
from pyspark.sql.types import ArrayType, FloatType
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pyspark.sql.functions import coalesce, array, col, lit
from pyspark.sql.functions import when, size, array


# ---------------------------------------
# 1) MiniLM Embedding 
# ---------------------------------------

def get_minilm_embeddings(text, use_gpu=False):
    """
    Generate embeddings for text using sentence-transformers/all-MiniLM-L6-v2.
    Handles lists by joining elements into a string.
    """
    if not hasattr(get_minilm_embeddings, "tokenizer"):
        get_minilm_embeddings.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        get_minilm_embeddings.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if use_gpu and torch.cuda.is_available():
            get_minilm_embeddings.model.to("cuda")

    tokenizer = get_minilm_embeddings.tokenizer
    model = get_minilm_embeddings.model

    # Convert lists to strings
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])  # Ensures lists are converted to a single text string

    if not text or not isinstance(text, str):
        return []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if use_gpu and torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()[0]

    return embeddings

# Register as UDF
minilm_embedding_udf = udf(get_minilm_embeddings, ArrayType(FloatType()))

# ---------------------------------------
# 2) Generate MiniLM Embeddings for Reviews
# ---------------------------------------

def generate_review_embeddings(df):
    """
    Generate MiniLM embeddings for review titles and texts.
    """
    df = df.withColumn("review_title_embedding", minilm_embedding_udf(col("review_title")))
    df = df.withColumn("review_text_embedding", minilm_embedding_udf(col("review_text")))
    return df

# ---------------------------------------
# 3) Combine Review Embeddings
# ---------------------------------------

def combine_review_embeddings(df):
    """
    Combine review title and text embeddings into a single embedding column.
    Uses default zero-array if missing.
    """
    df = df.withColumn(
        "combined_embedding",
        combine_all_embeddings_udf(
            col("review_title_embedding"),
            col("review_text_embedding"),
            array([lit(0.0)] * 384)  # ✅ Ensures a valid fallback
        )
    )
    return df

# ---------------------------------------
# 4) Generate Embeddings for Topics & Sentiments
# ---------------------------------------

from pyspark.sql.functions import expr, col, array, lit

def generate_topic_sentiment_embeddings(df):
    """
    Generate embeddings for extracted topics and sentiments.
    - Converts lists into strings before passing them to the embedding function.
    - Extracts `label` from `sentiment` (STRUCT).
    """
    # Extract sentiment labels (POSITIVE, NEGATIVE, etc.)
    df = df.withColumn("sentiment_labels", expr("transform(sentiment, s -> s.label)"))  
    
    # Convert arrays to space-separated strings
    df = df.withColumn("topic_text", expr("concat_ws(' ', topics)"))
    df = df.withColumn("sentiment_text", expr("concat_ws(' ', sentiment_labels)"))

    # Generate embeddings
    df = df.withColumn("topic_embedding", minilm_embedding_udf(col("topic_text")))
    df = df.withColumn("sentiment_embedding", minilm_embedding_udf(col("sentiment_text")))

    # Drop temporary columns
    df = df.drop("topic_text", "sentiment_text", "sentiment_labels")

    return df

# ---------------------------------------
# 5) Merge All Embeddings
# ---------------------------------------

import numpy as np

def combine_all_embeddings(embedding1, embedding2, embedding3):
    """
    Combine review, topic, and sentiment embeddings by averaging them safely.
    If all embeddings are missing, return a zero vector.
    """
    # Ensure each embedding is a valid array
    embedding1 = embedding1 if embedding1 is not None else []
    embedding2 = embedding2 if embedding2 is not None else []
    embedding3 = embedding3 if embedding3 is not None else []
    
    embeddings = [e for e in [embedding1, embedding2, embedding3] if len(e) > 0]

    if len(embeddings) == 0:
        return np.zeros(384).tolist()  # Ensure correct embedding size

    return np.mean(embeddings, axis=0).tolist()

combine_all_embeddings_udf = udf(combine_all_embeddings, ArrayType(FloatType()))

def merge_embeddings(df):
    """
    Merge all embeddings into a single column for model training.
    Ensures no `None` or empty lists cause issues.
    """
    from pyspark.sql.functions import coalesce, array, col, lit

    # Replace null values with empty arrays before merging
    df = df.withColumn("combined_embedding", coalesce(col("combined_embedding"), array([lit(0.0)] * 384)))
    df = df.withColumn("topic_embedding", coalesce(col("topic_embedding"), array([lit(0.0)] * 384)))
    df = df.withColumn("sentiment_embedding", coalesce(col("sentiment_embedding"), array([lit(0.0)] * 384)))


    # Merge embeddings into `final_embedding`
    df = df.withColumn(
        "final_embedding",
        combine_all_embeddings_udf(
            col("combined_embedding"),
            col("topic_embedding"),
            col("sentiment_embedding")
        )
    )
    return df

# ---------------------------------------
# 6) Validate & Clean Embeddings
# ---------------------------------------

def clean_aggregated_embeddings(df, embedding_col="final_embedding", expected_dim=384):
    """
    Validate and clean embeddings. If invalid, replace with zero-vector.
    """
    default_embedding = array([lit(0.0)] * expected_dim)

    return df.withColumn(
        embedding_col,
        when((col(embedding_col).isNotNull()) & (size(col(embedding_col)) == expected_dim), col(embedding_col))
        .otherwise(default_embedding)
    )