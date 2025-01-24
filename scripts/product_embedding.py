from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from transformers import AutoTokenizer, AutoModel
import torch
from pyspark.sql import functions as F

# -----------------------------
# 1) Embedding Generation
# -----------------------------

def get_minilm_embeddings(text, use_gpu=False):
    """
    Generate embeddings for the given text using sentence-transformers/all-MiniLM-L6-v2.
    """
    if not hasattr(get_minilm_embeddings, "tokenizer"):
        # Initialize tokenizer and model for each worker
        print("Initializing tokenizer and model in worker...")
        get_minilm_embeddings.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        get_minilm_embeddings.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if use_gpu and torch.cuda.is_available():
            get_minilm_embeddings.model.to("cuda")
        else:
            print("Using CPU as no GPU is available.")
    
    tokenizer = get_minilm_embeddings.tokenizer
    model = get_minilm_embeddings.model

    try:
        if not text or not isinstance(text, str):
            return []
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        if use_gpu and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()[0]
        
        return embeddings

    except Exception as e:
        print(f"Error generating embeddings for text: {text}. Error: {e}")
        return []

# Define PySpark UDF for embedding generation
minilm_embedding_udf = udf(get_minilm_embeddings, ArrayType(FloatType()))

def generate_review_embeddings(reviews_df, use_gpu=False):
    """
    Generate embeddings for review titles and texts in the reviews DataFrame.
    """
    reviews_df = reviews_df.fillna({
        "review_title_clean": "No title",
        "review_text_clean": "No review"
    })
    reviews_df = reviews_df.withColumn("review_title_embedding", minilm_embedding_udf(col("review_title_clean")))
    reviews_df = reviews_df.withColumn("review_text_embedding", minilm_embedding_udf(col("review_text_clean")))
    return reviews_df

# -----------------------------
# 2) Combine Embeddings
# -----------------------------

def combine_embeddings(embedding1, embedding2):
    """
    Combine two embeddings by averaging their values.
    """
    if embedding1 and embedding2:
        return [(e1 + e2) / 2 for e1, e2 in zip(embedding1, embedding2)]
    elif embedding1:
        return embedding1
    elif embedding2:
        return embedding2
    return []

combine_embeddings_udf = udf(combine_embeddings, ArrayType(FloatType()))

def combine_review_embeddings(reviews_df):
    """
    Combine review title and text embeddings into a single embedding column.
    """
    reviews_df = reviews_df.withColumn(
        "combined_embedding",
        combine_embeddings_udf(col("review_title_embedding"), col("review_text_embedding"))
    )
    return reviews_df

# -----------------------------
# 3) Aggregate Embeddings Across Multiple Reviews
# -----------------------------

def aggregate_embeddings_by_product(reviews_df):
    """
    Combine embeddings across multiple reviews for the same product by mean pooling.

    :param reviews_df: DataFrame with 'review_product_id' and 'combined_embedding' columns.
    :return: DataFrame with aggregated embeddings per product.
    """
    # Define a UDF to aggregate embeddings using mean pooling
    def mean_pooling(embeddings_list):
        if embeddings_list:
            # Calculate the mean of embeddings
            num_embeddings = len(embeddings_list)
            return [sum(vals) / num_embeddings for vals in zip(*embeddings_list)]
        return []

    mean_pooling_udf = F.udf(mean_pooling, ArrayType(FloatType()))

    # Group by product ID and aggregate embeddings
    aggregated_df = reviews_df.groupBy("review_product_id").agg(
        mean_pooling_udf(F.collect_list("combined_embedding")).alias("aggregated_embedding")
    )

    return aggregated_df