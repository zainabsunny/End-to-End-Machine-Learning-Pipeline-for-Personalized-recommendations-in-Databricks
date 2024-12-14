from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.functions import mean, stddev, col
from pyspark.sql.types import DoubleType, IntegerType
from transformers import AutoTokenizer, AutoModel
import torch

def transform_cosmetic_data(cosmetic_df):
    """Transform structured interaction data."""
    # Cache the DataFrame to ensure user_session doesn't get lost
    cosmetic_df = cosmetic_df.cache()
    
    # Scale `cosmetic_price`
    assembler = VectorAssembler(inputCols=["cosmetic_price"], outputCol="price_vec")
    cosmetic_df = assembler.transform(cosmetic_df)
    
    scaler = StandardScaler(inputCol="price_vec", outputCol="price_scaled")
    cosmetic_df = scaler.fit(cosmetic_df).transform(cosmetic_df)
    
    return cosmetic_df

def transform_reviews_data(reviews_df):
    """Transform unstructured sentiment data using GPT embeddings."""
    # Cast numerical columns to appropriate types
    reviews_df = reviews_df.withColumn("mrp", reviews_df["mrp"].cast(DoubleType())) \
        .withColumn("review_price", reviews_df["review_price"].cast(DoubleType())) \
        .withColumn("product_rating", reviews_df["product_rating"].cast(DoubleType())) \
        .withColumn("product_rating_count", reviews_df["product_rating_count"].cast(IntegerType()))
    
    # Example GPT-like model embeddings (modify for full integration)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
    
    def get_text_embeddings(text):
        """Generate embeddings for a given text using GPT-like model."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
    
    return reviews_df

def transform_mapping_data(mapping_df):
    """Transform product mapping data."""
    from pyspark.sql.functions import mean, stddev, col
    from pyspark.sql.types import IntegerType

    # Cast columns to IntegerType
    mapping_df = mapping_df.withColumn("cosmetic_id", mapping_df["cosmetic_id"].cast(IntegerType()))
    mapping_df = mapping_df.withColumn("review_id", mapping_df["review_id"].cast(IntegerType()))
    
    # Calculate mean and standard deviation for filtering outliers
    stats = mapping_df.select(
        mean("review_id").alias("mean_reviews"),
        stddev("review_id").alias("stddev_reviews")
    ).collect()[0]
    
    mean_reviews = stats["mean_reviews"]
    stddev_reviews = stats["stddev_reviews"]
    
    # Filter out outliers
    mapping_df = mapping_df.filter(
        (col("review_id") > mean_reviews - 3 * stddev_reviews) &
        (col("review_id") < mean_reviews + 3 * stddev_reviews)
    )
    
    return mapping_df 
