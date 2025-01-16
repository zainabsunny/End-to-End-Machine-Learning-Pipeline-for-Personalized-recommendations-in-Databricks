from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.functions import mean, stddev, col
from pyspark.sql.types import DoubleType, IntegerType, ArrayType, FloatType, StringType
from transformers import AutoTokenizer, AutoModel
import torch
from pyspark.sql.functions import udf, lit
import numpy as np

def transform_cosmetic_data(cosmetic_df):
    """Transform structured interaction data."""
    # Scale `cosmetic_price`
    assembler = VectorAssembler(inputCols=["cosmetic_price"], outputCol="price_vec")
    cosmetic_df = assembler.transform(cosmetic_df)
    
    scaler = StandardScaler(inputCol="price_vec", outputCol="price_scaled")
    cosmetic_df = scaler.fit(cosmetic_df).transform(cosmetic_df)
    
    return cosmetic_df

def transform_reviews_data(reviews_df, use_gpu=False):
    """Transform unstructured sentiment data using stella_en_400M_v5 embeddings."""
    # Cast numerical columns to appropriate types
    reviews_df = reviews_df.withColumn("mrp", reviews_df["mrp"].cast(DoubleType())) \
                           .withColumn("review_price", reviews_df["review_price"].cast(DoubleType())) \
                           .withColumn("product_rating", reviews_df["product_rating"].cast(DoubleType())) \
                           .withColumn("product_rating_count", reviews_df["product_rating_count"].cast(IntegerType()))
    
    # Load tokenizer and model for stella_en_400M_v5
    tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")
    model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5")

    # If you have a GPU cluster, you can optionally move the model to GPU right here
    if use_gpu:
        model.to("cuda")
    
    # Define padding token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def get_text_embeddings(text):
        """Generate embeddings for a given text using stella_en_400M_v5."""
        try:
            if text is None:
                return []
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to GPU if desired
            if use_gpu:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = model(**inputs)
            # Take the mean of the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings for text: {text}. Error: {e}")
            return []

    # Define UDF for embeddings
    embeddings_udf = udf(get_text_embeddings, ArrayType(FloatType()))

    # Generate embeddings for text columns
    reviews_df = reviews_df.withColumn("text_embedding", embeddings_udf(reviews_df["text"])) 

    return reviews_df

def transform_mapping_data(mapping_df):
    """Transform product mapping data."""
    from pyspark.sql.functions import mean, stddev, col
    from pyspark.sql.types import IntegerType

    # Cast columns to IntegerType
    mapping_df = mapping_df.withColumn("cosmetic_product_id", mapping_df["cosmetic_product_id"].cast(IntegerType()))
    mapping_df = mapping_df.withColumn("review_product_id", mapping_df["review_product_id"].cast(IntegerType()))
    
    # Calculate mean and standard deviation for filtering outliers
    stats = mapping_df.select(
        mean("review_product_id").alias("mean_reviews"),
        stddev("review_product_id").alias("stddev_reviews")
    ).collect()[0]
    
    mean_reviews = stats["mean_reviews"]
    stddev_reviews = stats["stddev_reviews"]
    
    # Filter out outliers
    mapping_df = mapping_df.filter(
        (col("review_product_id") > mean_reviews - 3 * stddev_reviews) &
        (col("review_product_id") < mean_reviews + 3 * stddev_reviews)
    )
    
    return mapping_df
